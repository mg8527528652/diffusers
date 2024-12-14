import os
import boto3
from botocore.client import Config

# Set up Wasabi connection
wasabi_s3 = boto3.client(
    's3',
    endpoint_url='https://s3.us-east-2.wasabisys.com',
    aws_access_key_id='8J824EFSZLNXXTRIDCIF',
    aws_secret_access_key='IwHDQrnL42iE1vo2Mvmez0YSennQXrrQXN2E4VpG',
    config=Config(signature_version='s3v4')
)

def upload_directory_to_s3(directory_path, bucket_name, s3_folder=''):
    """
    Uploads a directory to Wasabi S3.

    :param directory_path: Path to the local directory to upload
    :param bucket_name: S3 bucket name
    :param s3_folder: Folder in the bucket to upload the contents to (can be left empty for root)
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file == 'optimizer.bin':
                continue
            s3_key = os.path.join(s3_folder, os.path.relpath(file_path, directory_path)).replace("\\", "/")

            try:
                print(f'Uploading {file_path} to s3://{bucket_name}/{s3_key}')
                wasabi_s3.upload_file(file_path, bucket_name, s3_key)
            except Exception as e:
                print(f'Failed to upload {file_path}: {e}')

if __name__ == "__main__":
    # Define your Wasabi bucket name and local directory to upload
    bucket_name = 'ai-image-editor-webapp'
    directory_path = '/root/diffusers/examples/controlnet/controlnet-model'

    # Optional: define folder in S3 (leave empty to upload to bucket's root)
    s3_folder = 'mayank/cn_inpaint_sdxl_multi_channel_v4'

    upload_directory_to_s3(directory_path, bucket_name, s3_folder)
