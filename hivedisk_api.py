import requests
import logging
import os
import threading

# Constants (techno+dell@hivenet.com)
HIVE_USER_ID = "auth0|65cb4260e3fa438e319f5681"
HIVE_USER_KEY = "43fcca299710e4f6d032113d5cf8008dcd42ca0a4f4e399ac600fa728137a3fe"
TOKEN = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlpKU3prOWZmM2EyYWJjbzRRMm1jNCJ9.eyJpc3MiOiJodHRwczovL2Rldi02dGUxbHAweS51cy5hdXRoMC5jb20vIiwic3ViIjoiYXV0aDB8NjVjYjQyNjBlM2ZhNDM4ZTMxOWY1NjgxIiwiYXVkIjoiaHR0cHM6Ly9wbGF0Zm9ybS5wcmVwcm9kLmhpdmVuZXQuY29tLyIsImlhdCI6MTcxODg4NDQxNCwiZXhwIjoxNzE4OTcwODE0LCJzY29wZSI6Im9mZmxpbmVfYWNjZXNzIiwiYXpwIjoic1hDSGlJNG9kTjJrdjhQNDVTVmNRa3NLUnN5YkxKYVMiLCJwZXJtaXNzaW9ucyI6W119.oiTFYY-hUa1N4dAcLqDg20YSXxJpPfLMS9WJ5xPKyqcunPWVlBIZgxFB6Rf5_Kmj1rV2VLOc-yOZDZQVMgQpU_-QfS8ynxh6iYAOd1pYU1APycltobPD0Ph9c3unzLrNPxoHxbWpMoYLJe5NT4An2TDftwxrxzg5tw7Ub6V4ieyPM9BSxYAtnYnxDBW4iaXUKwen19quoxWgX5_k_5l2VqOsMnepewAOuzpCKDyh6Sjx3CPTrFaduKPKzQ4cCGyP3WRmWhZG0V_r51tBFut-bGBo7ljuaB2IXD2_EaivGGQxPuuy-V28GP-VrSr7IDktzrKNHpcUl5WWgNWPHykCDQ"

HIVE_WORKSPACE_MID = "3905ab00f8791714a906b6169d7575adb2f2fc81209d957c2d08094ca701757f"
HIVE_DEDUP_KEY = ""
HIVE_READ_KEY = "60a6b734c312011edf6a1c296d3cca9dba0420ce31e292b291200dcdcce5965d"
HIVE_TRASH_MID = "0be454a8cb939df073c7a2e6bc421a856cf10a76aa869a8bdac0fe918f13f9d8"


URL = "http://57.128.41.80:8585"
HEADERS = {
    "X-Hive-User-ID": HIVE_USER_ID,
    "X-Hive-User-Key": HIVE_USER_KEY,
    "X-Hive-Read-Key": HIVE_READ_KEY,
    "Authorization": TOKEN,
    "Content-Type": "application/json",
    "Accept": "application/json",
}

PATH = "./"
EXTENSIONS = [".txt", ".md", ".py", ".pdf", ".csv", ".xls", ".xlsx", ".docx", ".doc"]

logging.basicConfig(filename="hivedisk_api.log", level=logging.DEBUG)

# ====================================================

def list_workspace(mid=HIVE_WORKSPACE_MID):
    url = f"{URL}/workspaces/{mid}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 201:
        logging.error("Error requesting workspace mid %s" % mid)
        return {"directories": [], "files": []}
    data = response.json()
    files, directories = [], []
    volumes = data["volumes"]
    for volume in volumes:
        children = volume["children"]
        if children is None:
            continue
        for child in children:
            if child["kind"] == "directory":
                directories.append(child["mid"])
            else:
                files.append((child["name"], child["mid"]))
    return {"directories": directories, "files": files}

def list_directory(mid):
    url = f"{URL}/directories/{mid}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        logging.error("Error requesting directory mid %s" % mid)
        return {"directories": [], "files": []}
    data = response.json()
    files, directories = [], []
    children = data["children"]
    for child in children:
        if child["kind"] == "directory":
            directories.append(child["mid"])
        else:
            files.append((child["name"], child["mid"]))
    return {"directories": directories, "files": files}

def get_file(filename, mid, path=PATH):
    if os.path.isfile(path+filename):
        logging.debug("File %s already exists" % filename)
        return
    url = f"{URL}/files/{mid}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        logging.error("Error requesting file %s mid %s" % (filename, mid))
        return
    content_type = response.headers.get('Content-Type')
    if 'application/json' in content_type:
        logging.warning("Found json format in file %s mid %s" % (filename, mid))
        return
    elif 'application/octet-stream' in content_type:
        logging.debug("Downloading file %s mid %s [..]" % (filename, mid))
        binary_data = response.content
        with open(path+filename, "wb") as f:
            f.write(binary_data)
            logging.debug("Downloading file %s mid %s [OK]" % (filename, mid))
    return

def list_all_files(mid=HIVE_WORKSPACE_MID):
    files = []
    directories = []
    result = list_workspace(mid)
    directories.extend(result["directories"])
    files.extend(result["files"])
    while len(directories) > 0:
        mid = directories.pop(0)
        logging.debug("Browsing directory mid %s [..]" % mid)
        result = list_directory(mid)
        directories.extend(result["directories"])
        files.extend(result["files"])
        logging.debug("Browsing directory mid %s [OK]" % mid)
    return files

def get_files(filelist, path=PATH, extensions=EXTENSIONS, nthreads=8):
    def func(i, j):
        for (filename, mid) in filelist[i:j]:
            if os.path.splitext(filename)[1] not in extensions:
                logging.debug("File %s is not a valid format; ignoring" % filename)
                continue
            get_file(filename, mid, path)
    threads = []
    for t in range(nthreads):
        i = t * len(filelist)//nthreads
        j = min(len(filelist), (t+1) * len(filelist)//nthreads)
        thread = threading.Thread(target=func, args=(i, j,), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def get_files_legacy(filelist, path=PATH, extensions=EXTENSIONS):
    for (filename, mid) in filelist:
        if os.path.splitext(filename)[1] not in extensions:
            logging.debug("File %s is not a valid format; ignoring" % filename)
            continue
        get_file(filename, mid, path)


def main():
    print(list_all_files())

if __name__ == "__main__":
    main()

