import requests
import logging
import os
import threading

# Constants (techno@hivenet.com)
# HIVE_USER_ID = "auth0|6548c4dcd83e09ee3dff7be1"
# HIVE_USER_KEY = "1fcf54570098623d785d0c0fb12248ae3423212a741e670b42f12b4b9ba08280"
# TOKEN = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlpKU3prOWZmM2EyYWJjbzRRMm1jNCJ9.eyJpc3MiOiJodHRwczovL2Rldi02dGUxbHAweS51cy5hdXRoMC5jb20vIiwic3ViIjoiYXV0aDB8NjU1MWY3MDk2NWI5YWFlYzA1MjgwOWE0IiwiYXVkIjoiaHR0cHM6Ly9wbGF0Zm9ybS5wcmVwcm9kLmhpdmVuZXQuY29tLyIsImlhdCI6MTcwMDE2OTc3NywiZXhwIjoxNzAwMjU2MTc3LCJhenAiOiJzWENIaUk0b2ROMmt2OFA0NVNWY1Frc0tSc3liTEphUyIsInNjb3BlIjoib2ZmbGluZV9hY2Nlc3MiLCJwZXJtaXNzaW9ucyI6W119.cmATZyMuctCQfgjjkS9hgt5lAAgJPbwuyVgrVoBxUWNaJ_enKmggByy69vW3JImDDqeuRXhNbZIbCPAY2OVLSv53zMdx48A9qOYz9Qkjb84lJ-Bp5-vxWV56KTtRBkYPJ7LKQ-41TYZRgvmqq240REk2jSHVA3ZPadFNRyMGtuVKD4YcXZC_6m4XjgQ2D0F9mvVmtKIdPTF9XHq_2gd4H-a-okWQgufLBcHkNTloasXfN38CGsn_nmRbuvOMnCgCFkKnZeK3OZ7lhUojXkwyHOxwjI3S_3yZVnNTh2Y9EEil830mfQZEXyQmApqgTpKaVLcgaTws9_Fo_Zz_oq96KA"

# HIVE_WORKSPACE_MID = "5868a957244023911b34a1cea933e17d5e61a6a90c77b720c6bbc5c0216d98fe"
# HIVE_DEDUP_KEY = ""
# HIVE_READ_KEY = "fb62bbf8b439161b2ee9894591cc29b501a2f1343dc6b1ae1e10922657afb32f"
# HIVE_TRASH_MID = "0be454a8cb939df073c7a2e6bc421a856cf10a76aa869a8bdac0fe918f13f9d8"
# HIVE_VOLUME_MID = "15cdc804e020f4890c84f8c2fb11ba201d240c38e032a956a746564b0a31e230"

# Constants (techno+dell@hivenet.com)
HIVE_USER_ID = "auth0|65b946d4b96c06a825cbf707"
HIVE_USER_KEY = "ae60defbe1e8ffdac9529ea24d30116b1fa2e47917529bbff1296a5c6ed29643"
TOKEN = "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IlB1azc0d2o0VVhWYXN1Z2gyYndjOCJ9.eyJpc3MiOiJodHRwczovL2hpdmVuZXQtcHJvZC5ldS5hdXRoMC5jb20vIiwic3ViIjoiYXV0aDB8NjViOTQ2ZDRiOTZjMDZhODI1Y2JmNzA3IiwiYXVkIjoicGxhdGZvcm0uaGl2ZW5ldC5jb20iLCJpYXQiOjE3MDcyOTU2NzIsImV4cCI6MTcwNzM4MjA3MiwiYXpwIjoiMHZ5QlVKcGVTbnhaMDJBd0t3VjE5OHdBMERuYWNxVEwiLCJzY29wZSI6Im9mZmxpbmVfYWNjZXNzIn0.NEAWI1M94oYt5rFDDLVvAdOnT0NB-L-xaD_WeVkeQ0mXE2hKlowA2hUtvNwVawBzqeXXmTW0JXp1xjLEQzcW1woec0r3hpHdeiPR1eIWnuEqUVVkhgNGuSvyB6Ih3Nve3IkyPq0qSCR9KLeN3TOL76YTXQyv7pTXoC1_aW1fPE4uEXqouWDeIo3_ELKnA2Wl4aIwH0GVPRsTY7l4804H2X8Tnq6Pe53mRoiQhg03nPKgi00-Csai3PI3vjUb2Vzd_w8r3O331pmazylW0xJO4hHAPTk5HsuapS9I-KKdQLdlXjWxoqfrBrWSMCaQRc8KAYhEaE8WjaIH79IBblK06w"

HIVE_WORKSPACE_MID = "19265adb481b9bd287603798a5d8c3e0ac134c86fb16257bbfe2de1b20efe292"
HIVE_DEDUP_KEY = ""
HIVE_READ_KEY = "8eb4de7c3d48f99bd30b95f023e7f2b5ecad7de49d485bd533fbef7fa74bde22"
HIVE_TRASH_MID = "f74d181f6865939b2ec0ff130bfea943a5cd289249ea07f581b49f2e6957ac87"
HIVE_VOLUME_MID = "283eef7ebee3c2544e450dde97aaee4a297b8d8a70ddfd738c54994bc434cd4e"


URL = "http://57.128.41.80:8080"
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
        for child in volume["children"]:
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
