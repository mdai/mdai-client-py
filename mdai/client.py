import os
import sys
import threading
import json
import zipfile
import requests
import urllib3.exceptions
from retrying import retry
from tqdm import tqdm
import arrow
from .preprocess import Project


def retry_on_http_error(exception):
    valid_exceptions = [
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        urllib3.exceptions.HTTPError,
    ]
    return any([isinstance(exception, e) for e in valid_exceptions])


class Client:
    """Client for communicating with MD.ai backend API.
    Communication is via user access tokens (in MD.ai Hub, Settings -> User Access Tokens).
    """

    def __init__(self, domain="public.md.ai", access_token=None):
        if not domain.endswith((".md.ai")):
            raise ValueError("domain {} is invalid: should be format *.md.ai".format(domain))

        self.domain = domain
        self.access_token = access_token
        self.session = requests.Session()
        self._test_endpoint()

    def project(self, project_id, path=".", force_download=False, annotations_only=False):
        """Initializes Project class given project id.

        Arguments:
            project_id: hash ID of project.
            path: directory used for data.
        """
        if path == ".":
            print("Using working directory for data.")
        else:
            os.makedirs(path, exist_ok=True)
            print("Using path '{}' for data.".format(path))

        data_manager_kwargs = {
            "domain": self.domain,
            "project_id": project_id,
            "path": path,
            "session": self.session,
            "headers": self._create_headers(),
        }

        annotations_data_manager = ProjectDataManager(
            "annotations", force_download, **data_manager_kwargs
        )

        if not annotations_only:
            images_data_manager = ProjectDataManager(
                "images", force_download, **data_manager_kwargs
            )

        annotations_data_manager.create_data_export_job()

        if not annotations_only:
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

    def _create_headers(self):
        headers = {}
        if self.access_token:
            headers["x-access-token"] = self.access_token
        return headers

    def _test_endpoint(self):
        """Checks endpoint for validity and authorization.
        """
        test_endpoint = "https://{}/api/test".format(self.domain)
        r = self.session.get(test_endpoint, headers=self._create_headers())
        if r.status_code == 200:
            print("Successfully authenticated to {}.".format(self.domain))
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
        gql_endpoint = "https://{}/api/graphql".format(self.domain)
        headers = self._create_headers()
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"

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
        type,
        force_download,
        domain=None,
        project_id=None,
        path=".",
        session=None,
        headers=None,
    ):
        if type not in ["images", "annotations"]:
            raise ValueError("type must be 'images' or 'annotations'.")
        if not domain:
            raise ValueError("domain is not specified.")
        if not project_id:
            raise ValueError("project_id is not specified.")
        if not os.path.exists(path):
            raise OSError("Path '{}' does not exist.".format(path))

        self.type = type
        self.force_download = force_download

        self.domain = domain
        self.project_id = project_id
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
        endpoint = "https://{}/api/data-export/{}".format(self.domain, self.type)
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        if r.status_code == 202:
            msg = "Preparing {} export for project {}...".format(self.type, self.project_id)
            print(msg.ljust(100))
            self._check_data_export_job_progress()
        else:
            if r.status_code == 401:
                msg = (
                    "Project {} at domain {}".format(self.project_id, self.domain)
                    + " does not exist or you do not have sufficient permissions for access."
                )
                print(msg)
            self._on_data_export_job_error()

    def wait_until_ready(self):
        self._ready.wait()

    def _get_data_export_params(self):
        if self.type == "images":
            params = {"projectHashId": self.project_id, "exportFormat": "zip"}
        elif self.type == "annotations":
            # TODO: restrict to assigned labelgroup
            params = {
                "projectHashId": self.project_id,
                "labelGroupNum": None,
                "exportFormat": "json",
            }
        return params

    def _check_data_export_job_progress(self):
        """Poll for data export job progress.
        """
        endpoint = "https://{}/api/data-export/{}/progress".format(self.domain, self.type)
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)

        try:
            body = r.json()
            status = body["status"]
        except (TypeError, KeyError):
            self._on_data_export_job_error()
            return

        if status == "running":
            try:
                progress = int(body["progress"])
            except (TypeError, ValueError):
                progress = 0
            try:
                time_remaining = int(body["timeRemaining"])
            except (TypeError, ValueError):
                time_remaining = 0

            # print formatted progress info
            if progress > 0 and progress <= 100 and time_remaining > 0:
                if time_remaining > 45:
                    time_remaining_fmt = (
                        arrow.now().shift(seconds=time_remaining).humanize(only_distance=True)
                    )
                else:
                    # arrow humanizes <= 45 to 'in seconds' or 'just now',
                    # so we will opt to be explicit instead.
                    time_remaining_fmt = "in {} seconds".format(time_remaining)
                end_char = "\r" if progress < 100 else "\n"
                msg = "Exporting {} for project {}...{}% (time remaining: {}).".format(
                    self.type, self.project_id, progress, time_remaining_fmt
                )
                print(msg.ljust(100), end=end_char, flush=True)

            # run progress check at 1s intervals so long as status == 'running' and progress < 100
            if progress < 100:
                t = threading.Timer(1.0, self._check_data_export_job_progress)
                t.start()

            return
        elif status == "done":
            self._on_data_export_job_done()
        elif status == "error":
            self._on_data_export_job_error()

    def _on_data_export_job_done(self):
        endpoint = "https://{}/api/data-export/{}/done".format(self.domain, self.type)
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        try:
            file_key = r.json()["fileKey"]

            if file_key:

                # check cached file
                filepath = os.path.join(self.path, file_key)

                if self.type == "images":
                    data_path = os.path.splitext(filepath)[0]
                elif self.type == "annotations":
                    data_path = filepath

                # if file has been downloaded previously
                if not self.force_download:
                    if os.path.exists(filepath) or os.path.exists(data_path):
                        if self.type == "images":
                            if not os.path.exists(data_path):
                                print("Extracting archive: {}".format(file_key))
                                with zipfile.ZipFile(filepath, "r") as f:
                                    f.extractall(self.path)
                            self.data_path = data_path
                        elif self.type == "annotations":
                            self.data_path = data_path

                        # fire ready threading.Event
                        self._ready.set()
                        print(
                            "Using cached {} data for project {}.".format(
                                self.type, self.project_id
                            )
                        )
                    else:
                        # download in separate thread
                        t = threading.Thread(target=self._download_file, args=(file_key,))
                        t.start()
                else:
                    # download in separate thread
                    t = threading.Thread(target=self._download_file, args=(file_key,))
                    t.start()

        except (TypeError, KeyError):
            self._on_data_export_job_error(type, project_id)

    def _on_data_export_job_error(self):
        endpoint = "https://{}/api/data-export/{}/error".format(self.domain, self.type)
        params = self._get_data_export_params()
        r = self.session.post(endpoint, json=params, headers=self.headers)
        print("Error exporting {} for project {}.".format(self.type, self.project_id))

    def _download_file(self, file_key):
        """Downloads file via signed URL requested from MD.ai API.
        """
        print("Downloading file: {}".format(file_key))
        filepath = os.path.join(self.path, file_key)

        url = "https://{}/api/project-files/signedurl/get?key={}".format(
            self.domain, requests.utils.quote(file_key)
        )

        # stream response so we can display progress bar
        r = requests.get(url, stream=True, headers=self.headers)

        # total size in bytes
        total_size = int(r.headers.get("content-length", 0))
        block_size = 32 * 1024
        wrote = 0
        with open(filepath, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in r.iter_content(block_size):
                    f.write(chunk)
                    wrote = wrote + len(chunk)
                    pbar.update(block_size)
        if total_size != 0 and wrote != total_size:
            raise IOError("Error downloading file {}.".format(file_key))

        if self.type == "images":
            # unzip archive
            print("Extracting archive: {}".format(file_key))
            with zipfile.ZipFile(filepath, "r") as f:
                f.extractall(self.path)
            # images directory should have same name as archive file
            self.data_path = os.path.splitext(filepath)[0]
        elif self.type == "annotations":
            self.data_path = filepath

        # fire ready threading.Event
        self._ready.set()
        print("Success: {} data for project {} ready.".format(self.type, self.project_id))
