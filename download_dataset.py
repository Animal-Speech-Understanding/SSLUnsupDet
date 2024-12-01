from google.cloud import storage
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    
    # Download the file
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

def download_all_blobs(bucket_name, destination_dir):
    """Downloads all blobs from the bucket to the specified local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # List all blobs in the bucket
    blobs = bucket.list_blobs()
    
    for blob in blobs:
        # Construct a local path
        local_file_path = os.path.join(destination_dir, blob.name)
        
        # Download the blob
        download_blob(bucket_name, blob.name, local_file_path)

if __name__ == "__main__":
    # Specify your bucket name and the local directory to download to
    bucket_name = 'wmmsd-pmacro-bestof'
    destination_dir = '.'
    
    # Download all files from the bucket
    download_all_blobs(bucket_name, destination_dir)
