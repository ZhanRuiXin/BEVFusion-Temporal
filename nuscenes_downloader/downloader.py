import requests
import os
import hashlib
from tqdm import tqdm
import tarfile
import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# replace your email and password in https://www.nuscenes.org/
useremail = "<YOUR_EMAIL>"
password = "<YOUR_PASSWORD>"

output_dir = "./data/nuscenes"
region = 'asia'  # 'us' or 'asia'

# 并发下载数量
MAX_WORKERS = 8

download_files = {
    "v1.0-test_meta.tgz": "b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-test_blobs.tgz": "e065445b6019ecc15c70ad9d99c47b33",
    "v1.0-trainval01_blobs.tgz": "cbf32d2ea6996fc599b32f724e7ce8f2",
    "v1.0-trainval02_blobs.tgz": "aeecea4878ec3831d316b382bb2f72da",
    "v1.0-trainval03_blobs.tgz": "595c29528351060f94c935e3aaf7b995",
    "v1.0-trainval04_blobs.tgz": "b55eae9b4aa786b478858a3fc92fb72d",
    "v1.0-trainval05_blobs.tgz": "1c815ed607a11be7446dcd4ba0e71ed0",
    "v1.0-trainval06_blobs.tgz": "7273eeea36e712be290472859063a678",
    "v1.0-trainval07_blobs.tgz": "46674d2b2b852b7a857d2c9a87fc755f",
    "v1.0-trainval08_blobs.tgz": "37524bd4edee2ab99678909334313adf",
    "v1.0-trainval09_blobs.tgz": "a7fcd6d9c0934e4052005aa0b84615c0",
    "v1.0-trainval10_blobs.tgz": "31e795f2c13f62533c727119b822d739",
    "v1.0-trainval_meta.tgz": "537d3954ec34e5bcb89a35d4f6fb0d4a",
}


def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password
        },
        "ClientMetadata": {}
    })

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            print("Login successful!")
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)

    return None


def download_file_with_resume(url, save_file, md5, max_retries=3):
    """支持断点续传的下载函数"""
    
    # 检查文件是否已存在且完整
    if os.path.exists(save_file):
        print(f"{os.path.basename(save_file)} exists, checking MD5...")
        md5obj = hashlib.md5()
        with open(save_file, 'rb') as file:
            for chunk in iter(lambda: file.read(8192), b''):
                md5obj.update(chunk)
        if md5obj.hexdigest() == md5:
            print(f"✅ {os.path.basename(save_file)} already downloaded and verified")
            return save_file
        else:
            print(f"⚠️ MD5 mismatch, resuming download...")
    
    # 断点续传
    headers = {}
    downloaded_bytes = 0
    if os.path.exists(save_file):
        downloaded_bytes = os.path.getsize(save_file)
        headers['Range'] = f'bytes={downloaded_bytes}-'
        print(f"Resuming from {downloaded_bytes / (1024**3):.2f} GB")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            file_size = downloaded_bytes + int(response.headers.get('Content-Length', 0))
            
            # 使用 'ab' 模式追加写入
            mode = 'ab' if downloaded_bytes > 0 else 'wb'
            
            progress_bar = tqdm(
                total=file_size, 
                initial=downloaded_bytes,
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024,
                desc=os.path.basename(save_file), 
                ascii=True
            )
            
            md5obj = hashlib.md5()
            # 如果文件已存在部分，先计算已有部分的 MD5
            if downloaded_bytes > 0:
                with open(save_file, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        md5obj.update(chunk)
            
            with open(save_file, mode) as file:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        md5obj.update(chunk)
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            # 验证 MD5
            if md5obj.hexdigest() == md5:
                print(f"✅ {os.path.basename(save_file)} downloaded and verified")
                return save_file
            else:
                print(f"MD5 verification failed, retrying... (attempt {attempt + 1}/{max_retries})")
                # 删除损坏的文件
                if os.path.exists(save_file):
                    os.remove(save_file)
                downloaded_bytes = 0
                headers = {}
                
        except Exception as e:
            print(f"Error downloading {save_file}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            else:
                raise
    
    return None


def get_download_urls():
    """获取所有文件的下载链接"""
    print("Logging in...")
    bearer_token = login(useremail, password)
    if not bearer_token:
        return None
    
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }
    
    print("Getting download URLs...")
    download_data = {}
    
    for filename, md5 in download_files.items():
        api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={region}&project=nuScenes'
        
        for attempt in range(3):
            try:
                response = requests.get(api_url, headers=headers, timeout=30)
                if response.status_code == 200:
                    download_url = response.json()['url']
                    download_data[filename] = [download_url, os.path.join(output_dir, filename), md5]
                    print(f"✅ {filename}")
                    break
                else:
                    print(f"⚠️ Failed to get URL for {filename}: {response.status_code}")
                    if attempt < 2:
                        time.sleep(2)
            except Exception as e:
                print(f"⚠️ Error getting URL for {filename}: {e}")
                if attempt < 2:
                    time.sleep(5)
    
    return download_data


def main():
    print("Starting nuScenes downloader with multi-threading...")
    
    # 获取下载链接
    download_data = get_download_urls()
    if not download_data:
        print("Failed to get download URLs")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用线程池并发下载
    print(f"\nDownloading {len(download_data)} files with {MAX_WORKERS} concurrent downloads...")
    print("=" * 60)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for filename, (download_url, save_file, md5) in download_data.items():
            future = executor.submit(download_file_with_resume, download_url, save_file, md5)
            futures[future] = filename
        
        # 等待所有下载完成
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"✅ Completed: {filename}")
                else:
                    print(f"❌ Failed: {filename}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("All downloads completed!")
    print("Extracting files...")
    
    # 解压文件
    for filename, (_, save_file, _) in download_data.items():
        if os.path.exists(save_file):
            try:
                if filename.endswith(".tgz"):
                    print(f"📦 Extracting {filename}...")
                    with gzip.open(save_file, 'rb') as f_in:
                        with tarfile.open(fileobj=f_in, mode='r') as tar:
                            tar.extractall(output_dir)
                elif filename.endswith(".tar"):
                    print(f"📦 Extracting {filename}...")
                    with tarfile.open(save_file, 'r') as tar:
                        tar.extractall(output_dir)
                print(f"✅ Extracted: {filename}")
            except Exception as e:
                print(f"❌ Error extracting {filename}: {e}")
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()