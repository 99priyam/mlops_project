import pandas as pd
import pyodbc 
import mysql.connector as msc
import io
from io import BytesIO
from google.cloud import storage
import boto3
import json
import yaml
import os
import sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
import dill

class utility:
    def bucket(credentials_dict, bucket_name, file_name_path_or_object_key, cloud_name):
        if cloud_name.lower()=="gcp":
            storage_client = storage.Client.from_service_account_info(credentials_dict)
            BUCKET_NAME = bucket_name
            bucket = storage_client.get_bucket(BUCKET_NAME)
            filename = list(bucket.list_blobs(prefix=''))
            for name in filename:
                print(name.name)
            blob = bucket.blob(file_name_path_or_object_key)
            data = blob.download_as_string()
            df = pd.read_csv(io.BytesIO(data), encoding="utf-8", sep=",")
            return df

    def read_yaml_file(filename):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys)
