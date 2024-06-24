import requests
import logging
import os
import threading
import webbrowser
import time
import base64
import hashlib
import binascii
import sys
import json

URL = "http://57.128.41.80:8585"

PATH = "./"
EXTENSIONS = [".txt", ".md", ".py", ".pdf", ".csv", ".xls", ".xlsx", ".docx", ".doc"]

FILENAME = "technodell"

logging.basicConfig(filename="hivedisk_api.log", level=logging.DEBUG)

# ====================================================

ALGORITHM = "pbkdf2_sha512"

AUTH0_URL = "dev-6te1lp0y.us.auth0.com"
AUTH0_AUDIENCE = "https://platform.preprod.hivenet.com/"
AUTH0_CLIENT_ID = "sXCHiI4odN2kv8P45SVcQksKRsybLJaS"

class User:
    def __init__(self, passphrase=""):
        self.user_id = ""
        self.user_key = ""
        self.token = ""
        self.refresh_token = ""
        self.passphrase = passphrase
        self.read_key = ""
        self.workspace_mid = ""
        self._device_code = ""
        self.token_date = 0

    def _init_auth(self):
        auth_url = f'https://{AUTH0_URL}/oauth/device/code'
        headers = {
            'content-type': 'application/x-www-form-urlencoded',
        }
        data = {
            'client_id': AUTH0_CLIENT_ID,
            'scope': 'offline_access',
            'audience': AUTH0_AUDIENCE,
        }
        response = requests.post(auth_url, headers=headers, data=data)
        if response.status_code != 200:
            return
        data = response.json()
        verification_uri = data["verification_uri_complete"]
        device_code = data["device_code"]
        print(verification_uri)
        webbrowser.open(verification_uri)
        self._device_code = device_code

    def _poll_for_token(self):
        token_url = f'https://{AUTH0_URL}/oauth/token'
        headers = {
            'content-type': 'application/x-www-form-urlencoded',
        }
        data = {
            'client_id': AUTH0_CLIENT_ID,
            'device_code': self._device_code,
            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
        }
        response = requests.post(token_url, headers=headers, data=data)
        if response.status_code == 200:
            print(response.json())
            return response.json()
        else:
            return None


    def _get_user_id(self):
        token = self.token
        token_parts = token.split('.')
        token_part = token_parts[1]
        decoded_payload_bytes = base64.urlsafe_b64decode(token_part + '=' * (4 - len(token_part) % 4))
        decoded_payload_str = decoded_payload_bytes.decode('utf-8')
        decoded_payload_json = json.loads(decoded_payload_str)
        hive_user_id = decoded_payload_json.get('sub', None)
        return hive_user_id

    def _get_user_key(self):
        def aux_hash_password(password, salt, iterations=1000000):
            assert salt and isinstance(salt, str) and "$" not in salt
            assert isinstance(password, str)
            pw_hash = hashlib.pbkdf2_hmac("sha512", password.encode("utf-8"), salt.encode("utf-8"), iterations, 32)
            if sys.version_info < (3, 0):
                return(binascii.hexlify(pw_hash))
            else:
                return(binascii.hexlify(pw_hash).decode())
        
        def aux_verify_password(password, password_hash):
            if (password_hash or "").count("$") != 3:
                return False
            algorithm, iterations, salt, b64_hash = password_hash.split("$", 3)
            iterations = int(iterations)
            assert algorithm == ALGORITHM
            compare_hash = aux_hash_password(password, salt, iterations)
            return True

        return aux_hash_password(self.passphrase, self.user_id)

    def _init_workspace(self):
        headers = {
            'X-Hive-User-ID': self.user_id,
            'X-Hive-User-Key': self.user_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': "Bearer " + self.token,
        }
        url = f'{URL}/workspaces'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return
        data = response.json()
        self.workspace_mid = data["workspace_mid"]
        
        url = f'{URL}/workspaces/{self.workspace_mid}'
        response = requests.get(url, headers=headers)
        if response.status_code > 201:
            return
        data = response.json()
        self.read_key = data["read_key"]
    
    def authenticate(self):
        device_code = self._init_auth()
        data = None
        while data is None:
            time.sleep(5)
            data = self._poll_for_token()
        self.token = data["access_token"]
        self.token_date = time.time()
        self.refresh_token = data["refresh_token"]
        self.user_id = self._get_user_id()
        self.user_key = self._get_user_key()
        self._init_workspace()
        self.headers = {
            "X-Hive-User-ID": self.user_id,
            "X-Hive-User-Key": self.user_key,
            "X-Hive-Read-Key": self.read_key,
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def refresh_access_token(self):
        if self.refresh_token == "":
            return None
        if time.time() - self.token_date < 86400:
            return None
            
        token_url = f'https://{AUTH0_URL}/oauth/token'
        data = {
            'client_id': AUTH0_CLIENT_ID,
            # 'client_secret': client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token',
        }
        response = requests.post(token_url, data=data)
        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.token_date = time.time()
            return 1
        else:
            print('Error refreshing token:', response.status_code, response.text)
            return None

def load_user(filename):
    with open(filename, "r") as f:
        raw = f.read()
        user_dict = json.loads(raw)

    user = User()
    user.user_id = user_dict["user_id"]
    user.user_key = user_dict["user_key"]
    user.token = user_dict["token"]
    user.refresh_token = user_dict["refresh_token"]
    user.passphrase = user_dict["passphrase"]
    user.read_key = user_dict["read_key"]
    user.workspace_mid = user_dict["workspace_mid"]
    user._device_code = user_dict["_device_code"]
    user.token_date = user_dict["token_date"]
    user.headers = user_dict["headers"]

    return user

def save_user(user, filename):
    user_dict = user.__dict__
    raw = json.dumps(user_dict)
    with open(filename, "w") as f:
        f.write(raw)

# ====================================================

def list_workspace(user, mid=""):
    if mid == "":
        mid = user.workspace_mid
    headers = user.headers
    url = f"{URL}/workspaces/{mid}"
    response = requests.get(url, headers=headers)
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

def list_directory(user, mid):
    headers = user.headers
    url = f"{URL}/directories/{mid}"
    response = requests.get(url, headers=headers)
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

def get_file(user, filename, mid, path=PATH):
    headers = user.headers
    if os.path.isfile(path+filename):
        logging.debug("File %s already exists" % filename)
        return
    url = f"{URL}/files/{mid}"
    response = requests.get(url, headers=headers)
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

def list_all_files(user, mid=""):
    if mid == "":
        mid = user.workspace_mid
    headers = user.headers
    files = []
    directories = []
    result = list_workspace(user, mid)
    directories.extend(result["directories"])
    files.extend(result["files"])
    while len(directories) > 0:
        mid = directories.pop(0)
        logging.debug("Browsing directory mid %s [..]" % mid)
        result = list_directory(user, mid)
        directories.extend(result["directories"])
        files.extend(result["files"])
        logging.debug("Browsing directory mid %s [OK]" % mid)
    return files

def get_files(user, filelist, path=PATH, extensions=EXTENSIONS, nthreads=8):
    def func(i, j):
        for (filename, mid) in filelist[i:j]:
            if os.path.splitext(filename)[1] not in extensions:
                logging.debug("File %s is not a valid format; ignoring" % filename)
                continue
            get_file(user, filename, mid, path)
    threads = []
    for t in range(nthreads):
        i = t * len(filelist)//nthreads
        j = min(len(filelist), (t+1) * len(filelist)//nthreads)
        thread = threading.Thread(target=func, args=(i, j,), daemon=True)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

def get_files_legacy(user, filelist, path=PATH, extensions=EXTENSIONS):
    for (filename, mid) in filelist:
        if os.path.splitext(filename)[1] not in extensions:
            logging.debug("File %s is not a valid format; ignoring" % filename)
            continue
        get_file(user, filename, mid, path)

def main(filename=FILENAME):
    user = load_user(filename)
    if user.refresh_access_token() is not None:
        save_user(user, filename)
    files = list_all_files(user)
    return files

if __name__ == "__main__":
    assert len(sys.argv) > 1
    filename = sys.argv[1]
    print(main(filename))