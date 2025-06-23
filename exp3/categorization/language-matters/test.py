from google.cloud import storage

bucket_name = "exp3-gemini-batch-input"
destination_blob_name = "test_upload.txt"
source_file_content = "This is a test file to check GCS access."

# 创建客户端
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(destination_blob_name)

# 上传字符串为文件
blob.upload_from_string(source_file_content)

print(f"✅ Successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
