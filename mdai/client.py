import os
import threading
import re
import math
import uuid
import zipfile
import requests
import urllib3.exceptions
from retrying import retry
from tqdm import tqdm
import arrow
from .preprocess import Project
from . import __version__


def retry_on_http_error(exception):
    valid_exceptions = [
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        urllib3.exceptions.HTTPError,
    ]
    return any([isinstance(exception, e) for e in valid_exceptions])


ANNOTATIONS_IMPORT_DEFAULT_CHUNK_SIZE = 100000


class Client:
    """Client for communicating with MD.ai backend API.
    Communication is via user access tokens (in MD.ai Hub, Settings -> User Access Tokens).
    """

    def __init__(self, domain="public.md.ai", access_token=None):
        domain_match = re.match(r"^\w+\.md\.ai$", domain)
        dev_domain_match = re.match(r"^\w+\.mdai.dev(:\d+)?$", domain)
        if not domain_match and not dev_domain_match:
            raise ValueError(f"domain {domain} is invalid: should be format *.md.ai")

        self.domain = domain
        self.access_token = access_token
        self.session = requests.Session()
        self._test_endpoint()

    def project(
        self,
        project_id,
        dataset_id=None,
        label_group_id=None,
        path=".",
        force_download=False,
        annotations_only=False,
        extract_images=True,
    ):
        """Initializes Project class given project id.

        Arguments:
            project_id: hash ID of project
            dataset_id: hash ID of dataset to scope to (optional - default `None`)
            label_group_id: hash ID of the label group to scope to (optional - default `None`)
            path: directory used for data (optional - default `"."`)
            force_download: if `True`, ignores possible existing data in `path` (optional - default `False`)
            annotations_only: if `True`, downloads annotations only (optional - default `False`)
            extract_images: if 'True', automatically extracts downloaded zip files for image exports (optional - default 'True')
        """
        if path == ".":
            print("Using working directory for data.")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"Using path '{path}' for data.")

        data_manager_kwargs = {
            "domain": self.domain,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "label_group_id": label_group_id,
            "path": path,
            "session": self.session,
            "headers": self._create_headers(),
            "force_download": force_download,
            "extract_images": extract_images,
        }

        annotations_data_manager = ProjectDataManager("annotations", **data_manager_kwargs)
        annotations_data_manager.create_data_export_job()
        if not annotations_only:
            images_data_manager = ProjectDataManager("images", **data_manager_kwargs)
            images_data_manager.create_data_export_job()

        annotations_data_manager.wait_until_ready()
        if not annotations_only:
            images_data_manager.wait_until_ready()
            p = Project(
                annotations_fp=annotations_data_manager.data_path,
                images_dir=images_data_manager.data_path,
            )
            return p
        else:
            print("No project created. Downloaded annotations only.")
            return None

    def download_model_outputs(
        self, project_id, dataset_id=None, model_id=None, path=".", force_download=False
    ):
        """Downloads model outputs given project_id.

        Arguments:
            project_id: hash ID of project
            dataset_id: hash ID of dataset (optional - default `None`)
            model_id: hash ID of the model (optional - default `None`)
            path: directory used for data (optional - default `"."`)
            force_download: if `True`, ignores possible existing data in `path` (optional - default `False`)
        """
        if path == ".":
            print("Using working directory for model outputs.")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"Using path '{path}' for data.")

        data_manager_kwargs = {
            "domain": self.domain,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "path": path,
            "session": self.session,
            "headers": self._create_headers(),
            "force_download": force_download,
        }

        model_outputs_manager = ProjectDataManager("model-outputs", **data_manager_kwargs)
        model_outputs_manager.create_data_export_job()
        model_outputs_manager.wait_until_ready()
        return None

    def download_dicom_metadata(
        self, project_id, dataset_id=None, format="json", path=".", force_download=False
    ):
        """Downloads dicom metadata given project_id, dataset_id and export format.

        Arguments:
            project_id: hash ID of project
            dataset_id: hash ID of dataset (optional - default `None`)
            format: export format for the metadata file, json or csv (optional - default `json`)
            path: directory used for data (optional - default `"."`)
            force_download: if `True`, ignores possible existing data in `path` (optional - default `False`)
        """
        if path == ".":
            print("Using working directory for dicom metadata.")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"Using path '{path}' for data.")

        if format not in ["json", "csv"]:
            raise Exception(
                "Incorrect export format specified for dicom-metadata. Only json and csv formats are supported."
            )

        data_manager_kwargs = {
            "domain": self.domain,
            "project_id": project_id,
            "dataset_id": dataset_id,
            "format": format,
            "path": path,
            "session": self.session,
            "headers": self._create_headers(),
            "force_download": force_download,
        }

        dicom_metadata_manager = ProjectDataManager("dicom-metadata", **data_manager_kwargs)
        dicom_metadata_manager.create_data_export_job()
        dicom_metadata_manager.wait_until_ready()
        return None

    def load_model_annotations(self):
        """Deprecated method: use `import_annotations` instead.
        """
        print("Deprecated method: use `import_annotations` instead.")

    def import_annotations(
        self, annotations, project_id, dataset_id, chunk_size=ANNOTATIONS_IMPORT_DEFAULT_CHUNK_SIZE,
    ):
        """Import annotations into project.
        For example, this method can be used to load machine learning model results into project as
        annotations, or quickly populate metadata labels.

        Arguments:
            project_id: hash ID of project.
            dataset_id: hash ID of dataset.
            annotations: list of annotations to load.
            chunk_size: number of annotations to load as a chunk.
        """
        if not annotations:
            print("No annotations provided.")
        if not project_id:
            print("project_id is required.")
        if not dataset_id:
            print("dataset_id is required.")

        num_chunks = math.ceil(len(annotations) / chunk_size)

        if num_chunks > 1:
            print(f"Importing {len(annotations)} total annotations in {num_chunks} chunks...")

        failed_annotations = []

        for i in range(num_chunks):
            if num_chunks > 1:
                print(f"Chunk {i+1}...")

            start = i * chunk_size
            end = (i + 1) * chunk_size
            annotations_chunk = annotations[start:end]

            manager = AnnotationsImportManager(
                annotations=annotations_chunk,
                project_id=project_id,
                dataset_id=dataset_id,
                session=self.session,
                domain=self.domain,
                headers=self._create_headers(),
            )
            manager.create_job()
            manager.wait_until_ready()

            for failed_annotation in manager.failed_annotations:
                # add start index since returned index is for chunk
                failed_annotation["index"] += start
                failed_annotations.append(failed_annotation)

        if num_chunks > 1:
            num_failed = len(failed_annotations)
            print(
                f"Successfully imported {len(annotations) - num_failed} / {len(annotations)}"
                + f" total annotations into project {project_id}."
            )

        return failed_annotations

    def _create_headers(self):
        headers = {}
        if self.access_token:
            headers["x-access-token"] = self.access_token
        return headers

    def _test_endpoint(self):
        """Checks endpoint for validity and authorization.
        """
        test_endpoint = f"https://{self.domain}/api/test"
        r = self.session.get(test_endpoint, headers=self._create_headers())
        if r.status_code == 200:
            print(f"Successfully authenticated to {self.domain}.")
        else:
            raise Exception("Authorization error. Make sure your access token is valid.")

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _gql(self, query, variables=None):
        """Executes GraphQL query.
        """
        gql_endpoint = f"https://{self.domain}/api/graphql"
        headers = self._create_headers()
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"
        headers["apollographql-client-name"] = ("mdai-client-py",)
        headers["apollographql-client-version"] = __version__

        data = {"query": query, "variables": variables}
        r = self.session.post(gql_endpoint, headers=headers, json=data)
        if r.status_code != 200:
            r.raise_for_status()

        body = r.json()
        data = body["data"] if "data" in body else None
        errors = body["errors"] if "errors" in body else None

        return data, errors


class ProjectDataManager:
    """Manager for project data exports and downloads.
    """

    def __init__(
        self,
        data_type,
        domain=None,
        project_id=None,
        dataset_id=None,
        label_group_id=None,
        model_id=None,
        format=None,
        path=".",
        session=None,
        headers=None,
        force_download=False,
        extract_images=True,
    ):
        if data_type not in ["images", "annotations", "model-outputs", "dicom-metadata"]:
            raise ValueError(
                "data_type must be 'images', 'annotations', 'model-outputs' or 'dicom-metadata'."
            )
        if not domain:
            raise ValueError("domain is not specified.")
        if not project_id:
            raise ValueError("project_id is not specified.")
        if not os.path.exists(path):
            raise OSError(f"Path '{path}' does not exist.")

        self.data_type = data_type
        self.force_download = force_download
        self.extract_images = extract_images

        self.domain = domain
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.label_group_id = label_group_id
        self.format = format
        self.model_id = model_id
        self.path = path
        if session and isinstance(session, requests.Session):
            self.session = session
        else:
            self.session = requests.Session()
        self.headers = headers

        # path for downloaded data
        self.data_path = None
        # ready threading event
        self._ready = threading.Event()

    def create_data_export_job(self):
        """Create data export job through MD.ai API.
        This is an async operation. Status code of 202 indicates successful creation of job.
        """
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code == 202:
            msg = f"Preparing {self.data_type} export for project {self.project_id}..."
            print(msg.ljust(100))
            self._check_data_export_job_progress()
        else:
            if r.status_code == 401:
                msg = (
                    f"Project {self.project_id} at domain {self.domain}"
                    + " does not exist or you do not have sufficient permissions for access."
                )
                print(msg)
            self._on_data_export_job_error()

    def wait_until_ready(self):
        self._ready.wait()

    def _get_data_export_params(self):
        if self.data_type == "images":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "exportFormat": "zip",
            }
        elif self.data_type == "annotations":
            # TODO: restrict to assigned labelgroup
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "labelGroupHashId": self.label_group_id,
                "exportFormat": "json",
            }
        elif self.data_type == "model-outputs":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "modelHashId": self.model_id,
                "exportFormat": "json",
            }
        elif self.data_type == "dicom-metadata":
            params = {
                "projectHashId": self.project_id,
                "datasetHashId": self.dataset_id,
                "exportFormat": self.format,
            }
        return params

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _check_data_export_job_progress(self):
        """Poll for data export job progress.
        """
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/progress"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            body = r.json()
            status = body["status"]
        except (TypeError, KeyError):
            self._on_data_export_job_error()
            return

        if status == "done":
            self._on_data_export_job_done()

        elif status == "error":
            self._on_data_export_job_error()

        elif status == "running":
            try:
                progress = int(body["progress"])
            except (TypeError, ValueError):
                progress = 0
            try:
                time_remaining = int(body["timeRemaining"])
            except (TypeError, ValueError):
                time_remaining = 0

            # print formatted progress info
            if time_remaining > 45:
                time_remaining_fmt = (
                    arrow.now().shift(seconds=time_remaining).humanize(only_distance=True)
                )
            else:
                # arrow humanizes <= 45 to 'in seconds' or 'just now',
                # so we will opt to be explicit instead.
                time_remaining_fmt = f"{time_remaining} seconds"
            end_char = "\r" if progress < 100 else "\n"
            msg = (
                f"Exporting {self.data_type} for project {self.project_id}..."
                + f"{progress}% (time remaining: {time_remaining_fmt})."
            )
            print(msg.ljust(100), end=end_char, flush=True)

            # run progress check at 1s intervals so long as status == 'running'
            t = threading.Timer(1.0, self._check_data_export_job_progress)
            t.start()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_data_export_job_done(self):
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/done"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            file_keys = r.json()["fileKeys"]

            if file_keys:
                data_path = self._get_data_path(file_keys)
                if self.force_download or not os.path.exists(data_path):
                    # download in separate thread
                    t = threading.Thread(target=self._download_files, args=(file_keys,))
                    t.start()
                else:
                    # use existing data
                    self.data_path = data_path
                    print(f"Using cached {self.data_type} data for project {self.project_id}.")
                    # fire ready threading.Event
                    self._ready.set()
        except (TypeError, KeyError):
            self._on_data_export_job_error()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_data_export_job_error(self):
        endpoint = f"https://{self.domain}/api/data-export/{self.data_type}/error"
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()
        print(f"Error exporting {self.data_type} for project {self.project_id}.")
        # fire ready threading.Event
        self._ready.set()

    def _get_data_path(self, file_keys):
        if self.data_type == "images":
            # should be folder for zip file:
            # xxxx.zip -> xxxx/
            # xxxx_part1of3.zip -> xxxx/
            images_dir = re.sub(r"(_part\d+of\d+)?\.\S+$", "", file_keys[0])
            return os.path.join(self.path, images_dir)
        elif self.data_type == "annotations":
            # annotations export will be single file
            annotations_fp = file_keys[0]
            return os.path.join(self.path, annotations_fp)
        elif self.data_type == "model-outputs":
            # model outputs export will be single file
            model_outputs_fp = file_keys[0]
            return os.path.join(self.path, model_outputs_fp)
        elif self.data_type == "dicom-metadata":
            # dicom metadata export will be single file
            dicom_metadata_fp = file_keys[0]
            return os.path.join(self.path, dicom_metadata_fp)

    def _download_files(self, file_keys):
        """Downloads exported files.
        """
        try:
            for file_key in file_keys:
                print(f"Downloading file: {file_key}")
                filepath = os.path.join(self.path, file_key)

                key = requests.utils.quote(file_key)
                dl_session_id = str(uuid.uuid4())

                # request download token
                url = f"https://{self.domain}/api/data-export/download-request"
                data = {"key": key, "sessionId": dl_session_id}
                r = requests.post(url, json=data, headers=self.headers)
                dl_token = r.json().get("token")

                # download file
                # stream response so we can display progress bar
                params = {"token": dl_token, "sessionId": dl_session_id}
                url = f"https://{self.domain}/api/data-export/download/{key}"
                r = requests.get(url, params=params, headers=self.headers, stream=True)

                # total size in bytes
                total_size = int(r.headers.get("content-length", 0))
                block_size = 32 * 1024
                wrote = 0
                with open(filepath, "wb") as f:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, unit_divisor=1024
                    ) as pbar:
                        for chunk in r.iter_content(block_size):
                            f.write(chunk)
                            wrote = wrote + len(chunk)
                            pbar.update(block_size)
                if total_size != 0 and wrote != total_size:
                    raise IOError(f"Error downloading file {file_key}.")

                if self.data_type == "images" and self.extract_images:
                    # unzip archive
                    print(f"Extracting archive: {file_key}")
                    with zipfile.ZipFile(filepath, "r") as f:
                        f.extractall(self.path)

            self.data_path = self._get_data_path(file_keys)

            print(f"Success: {self.data_type} data for project {self.project_id} ready.")
        except Exception:
            print(f"Error downloading {self.data_type} data for project {self.project_id}.")

        # fire ready threading.Event
        self._ready.set()


class AnnotationsImportManager:
    """Manager for importing annotations.
    """

    def __init__(
        self,
        annotations=None,
        project_id=None,
        dataset_id=None,
        session=None,
        domain=None,
        headers=None,
    ):
        if not domain:
            raise ValueError("domain is not specified.")
        if not project_id:
            raise ValueError("project_id is not specified.")

        self.annotations = annotations
        self.project_id = project_id
        self.dataset_id = dataset_id
        if session and isinstance(session, requests.Session):
            self.session = session
        else:
            self.session = requests.Session()
        self.domain = domain
        self.headers = headers

        self.job_id = None

        # list of failed annotation imports
        self.failed_annotations = []

        # ready threading event
        self._ready = threading.Event()

    def create_job(self):
        """Create annotations import job through MD.ai API.
        This is an async operation. Status code of 202 indicates successful creation of job.
        """
        endpoint = f"https://{self.domain}/api/data-import/annotations"
        params = {
            "projectHashId": self.project_id,
            "datasetHashId": self.dataset_id,
            "annotations": self.annotations,
        }

        # reset list of failed annotation imports
        self.failed_annotations = []

        r = self.session.post(endpoint, json=params, headers=self.headers)

        if r.status_code == 202:
            self.job_id = r.json()["jobId"]
            msg = f"Importing {len(self.annotations)} annotations into "
            msg += f"project {self.project_id}, "
            msg += f"dataset {self.dataset_id}..."
            print(msg.ljust(100))
            self._check_job_progress()
        else:
            print(r.status_code)
            if r.status_code in (400, 401):
                msg = "Provided IDs are invalid, or you do not have sufficient permissions."
                print(msg)
            self._on_job_error()

    def wait_until_ready(self):
        self._ready.wait()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _check_job_progress(self):
        """Poll for annotations import job progress.
        """
        endpoint = f"https://{self.domain}/api/data-import/annotations/progress"
        params = {"projectHashId": self.project_id, "jobId": self.job_id}
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            body = r.json()
            status = body["status"]
        except (TypeError, KeyError):
            self._on_job_error()
            return

        if status == "done":
            self._on_job_done()

        elif status == "error":
            self._on_job_error()

        elif status == "running":
            try:
                progress = int(body["progress"])
            except (TypeError, ValueError):
                progress = 0
            try:
                time_remaining = int(body["timeRemaining"])
            except (TypeError, ValueError):
                time_remaining = 0

            # print formatted progress info
            if time_remaining > 45:
                time_remaining_fmt = (
                    arrow.now().shift(seconds=time_remaining).humanize(only_distance=True)
                )
            else:
                # arrow humanizes <= 45 to 'in seconds' or 'just now',
                # so we will opt to be explicit instead.
                time_remaining_fmt = f"{time_remaining} seconds"
            end_char = "\r" if progress < 100 else "\n"
            msg = (
                f"Annotations import for project {self.project_id}..."
                + f"{progress}% (time remaining: {time_remaining_fmt})."
            )
            print(msg.ljust(100), end=end_char, flush=True)

            # run progress check at 1s intervals so long as status == 'running'
            t = threading.Timer(1.0, self._check_job_progress)
            t.start()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_job_done(self):
        endpoint = f"https://{self.domain}/api/data-import/annotations/done"
        params = {"projectHashId": self.project_id, "jobId": self.job_id}
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()

        try:
            body = r.json()
            self.failed_annotations = body["failed"]
        except (TypeError, KeyError):
            self._on_job_error()
            return

        num_failed = len(self.failed_annotations)
        print(
            f"Successfully imported {len(self.annotations) - num_failed} / {len(self.annotations)}"
            + f" annotations into project {self.project_id}"
            + f", dataset {self.dataset_id}."
        )
        # fire ready threading.Event
        self._ready.set()

    @retry(
        retry_on_exception=retry_on_http_error,
        wait_exponential_multiplier=100,
        wait_exponential_max=1000,
        stop_max_attempt_number=10,
    )
    def _on_job_error(self):
        endpoint = f"https://{self.domain}/api/data-import/annotations/error"
        params = {"projectHashId": self.project_id, "jobId": self.job_id}
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code != 200:
            r.raise_for_status()
        print(
            f"Error importing annotations into project {self.project_id}"
            + f", dataset {self.dataset_id}."
        )
        # fire ready threading.Event
        self._ready.set()
