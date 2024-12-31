from __future__ import annotations

import json
import shlex
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.project_project_openshift_io import Project
from ocp_resources.project_request import ProjectRequest
from ocp_resources.role import Role
from ocp_resources.secret import Secret
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger

from tests.model_serving.model_server.utils import b64_encoded_string
from utilities.general import get_s3_secret_dict


LOGGER = get_logger(name=__name__)
TIMEOUT_2MIN = 2 * 60


@contextmanager
def create_ns(
    name: str,
    admin_client: Optional[DynamicClient] = None,
    unprivileged_client: Optional[DynamicClient] = None,
    teardown: bool = True,
    delete_timeout: int = 4 * 60,
    labels: Optional[Dict[str, str]] = None,
) -> Generator[Namespace, None, None]:
    if unprivileged_client:
        with ProjectRequest(name=name, client=unprivileged_client, teardown=teardown):
            project = Project(
                name=name,
                client=unprivileged_client,
                teardown=teardown,
                delete_timeout=delete_timeout,
            )
            project.wait_for_status(project.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield project

    else:
        with Namespace(
            client=admin_client,
            name=name,
            label=labels,
            teardown=teardown,
            delete_timeout=delete_timeout,
        ) as ns:
            ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=TIMEOUT_2MIN)
            yield ns


def wait_for_kserve_predictor_deployment_replicas(client: DynamicClient, isvc: InferenceService) -> Deployment:
    ns = isvc.namespace

    deployments = list(
        Deployment.get(
            label_selector=f"{isvc.ApiGroup.SERVING_KSERVE_IO}/inferenceservice={isvc.name}",
            client=client,
            namespace=isvc.namespace,
        )
    )

    if len(deployments) == 1:
        deployment = deployments[0]
        if deployment.exists:
            deployment.wait_for_replicas()
            return deployment

    elif len(deployments) > 1:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {ns}")

    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {ns}")


@contextmanager
def s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, None, None]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={"opendatahub.io/connection-type": "s3"},
        data_dict=get_s3_secret_dict(
            aws_access_key=aws_access_key,
            aws_secret_access_key=aws_secret_access_key,
            aws_s3_bucket=aws_s3_bucket,
            aws_s3_endpoint=aws_s3_endpoint,
            aws_s3_region=aws_s3_region,
        ),
        wait_for_resource=True,
    ) as secret:
        yield secret


@contextmanager
def create_storage_config_secret(
    admin_client: DynamicClient,
    endpoint_secret_name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_bucket: str,
    aws_s3_region: str,
    aws_s3_endpoint: str,
) -> Generator[Secret, None, None]:
    secret = {
        "access_key_id": aws_access_key,
        "bucket": aws_s3_bucket,
        "default_bucket": aws_s3_bucket,
        "endpoint_url": aws_s3_endpoint,
        "region": aws_s3_region,
        "secret_access_key": aws_secret_access_key,
        "type": "s3",
    }
    data = {endpoint_secret_name: b64_encoded_string(string_to_encode=json.dumps(secret))}
    with Secret(
        client=admin_client,
        namespace=namespace,
        data_dict=data,
        wait_for_resource=True,
        name="storage-config",
    ) as storage_config:
        yield storage_config


@contextmanager
def create_isvc_view_role(
    client: DynamicClient,
    isvc: InferenceService,
    name: str,
    resource_names: Optional[List[str]] = None,
) -> Role:
    rules = [
        {
            "apiGroups": [isvc.api_group],
            "resources": ["inferenceservices"],
            "verbs": ["get"],
        },
    ]

    if resource_names:
        rules[0].update({"resourceNames": resource_names})

    with Role(
        client=client,
        name=name,
        namespace=isvc.namespace,
        rules=rules,
    ) as role:
        yield role


def login_with_user_password(api_address: str, user: str, password: str | None = None) -> bool:
    """
    Log in to an OpenShift cluster using a username and password.

    Args:
        api_address (str): The API address of the OpenShift cluster.
        user (str): Cluster's username
        password (str, optional): Cluster's password

    Returns:
        bool: True if login is successful otherwise False.
    """
    login_command: str = f"oc login  --insecure-skip-tls-verify=true {api_address} -u {user}"
    if password:
        login_command += f" -p '{password}'"

    _, out, _ = run_command(command=shlex.split(login_command))

    return "Login successful" in out
