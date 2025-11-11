import os
import time
import asyncio
from typing import Tuple
from datetime import datetime,timezone, timedelta
from azure.storage.blob import BlobServiceClient, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError, ResourceModifiedError
from azure.data.tables import TableClient, TableServiceClient, UpdateMode
from typing_extensions import TypedDict
from dotenv import find_dotenv, load_dotenv

from .pubsub_manager import PubSubManager
class RootDataTable(TypedDict, total=False):
    PartitionKey: str           # ID                    # Passed to Container
    RowKey: str                 # Filename              # Passed to Container
    ext: str                    # File Extension
    file_size_gb: float         # X GB
    file_upload_full_path: str  # Blob pcdUploads/filename.laz
    process_folder:str          # Blob processed/foldername
    log_root: str               # Blob logs/log_root.csv
    log_file: str               # Blob logs/log_filename.csv 
    upload_completed:str        # [True, False]             
    process_completed:str       # [True, False]
    store_completed:str         # [True, False]
    status:str                  # [upload, process, store, completed, replaced] 
    # Edited by Container or Function App
    coordinates:str
    upload_starttime:str
    process_starttime:str       # 
    process_expiretime:str      #
    replaced:str                # [True, False]
    completed:str               # [True, False]        
    error:str                   # [True, False]
    error_msg:str               # [Upload not completed]
    # Function App
    trees_completed : int       # [Number of Trees]
    
class Blob_Manager():
    def __init__(self, conn_string, container_name):
        self.conn_string = conn_string
        self.container_name = container_name
        self.blob_service = BlobServiceClient.from_connection_string(conn_string)
        self.blob_container_client = self._check_container_exist(container_name)
        self.blob_account_key = self.blob_service.credential.account_key
        self.blob_account_name = self.blob_service.account_name
        self.blob_primary_endpoint = self.blob_service.primary_endpoint
        
    def _check_container_exist(self, container_name):
        # ✅ Ensure container exists, if not exist create one
        container = self.blob_service.get_container_client(container_name)
        try:
            container.get_container_properties()
        except ResourceNotFoundError:
            container.create_container()
        return container
    
    def download_file(self, blob_path, write_path):
        blob_file_client = self.blob_container_client.get_blob_client(blob=blob_path)
        with open(file=write_path, mode="wb") as file_download:
            download_stream = blob_file_client.download_blob()
            file_download.write(download_stream.readall())
    
    def upload_file(self, docker_path, write_path):
        blob_file_client = self.blob_container_client.get_blob_client(blob=write_path)
        with open(file=docker_path, mode="rb") as data:
            blob_file_client.upload_blob(data=data, overwrite=True)

class DBManager(PubSubManager):
    def __init__(self):
        self.init_pubsub()
        load_dotenv(find_dotenv())
        # Blob Init
        self.strg_account_name    = str(os.environ["StorageAccName"])
        self.strg_access_key      = str(os.environ["StorageAccKey"])
        self.strg_endpoint_suffix = str(os.environ["StorageEndpointSuffix"])
        self.storageContainer   = str(os.getenv("StorageContainer"))
        self.connection_string      = f"DefaultEndpointsProtocol=https;AccountName={self.strg_account_name};AccountKey={self.strg_access_key};EndpointSuffix={self.strg_endpoint_suffix}"
        self.blob_obj = Blob_Manager(self.connection_string, self.storageContainer)
        
        # Logs
        self.root_log_table_name= str(os.getenv("DBRoot", "rootLog"))
        self.PartitionKey = str(os.getenv("PartitionKey"))
        self.row_key = self.filename = str(os.getenv("RowKey"))
        
        # Download
        self.download_full_path = str(os.getenv("file_upload_full_path"))
        self.download_file_extension = str(os.getenv("ext"))
        
        # Upload
        self.process_folder = str(os.getenv("process_folder"))
        self.data_loc = str(os.getenv("DATA_LOC"))
        
        
        # Docker in out
        self.docker_input_folder  = f"/root/data_in"
        self.docker_output_folder = f"/root/data_out"
        
        # self.create_database_if_not_exists()
        print(f"\n\n\n\
            PartitionKey : {self.PartitionKey}\n\
            Storage_container : {self.storageContainer}\n\
            DownloadFullPath : {self.download_full_path}\n\
            Extention : {self.download_file_extension}\
            ")
        
        # self.upload_everything("/app")
        
    def download_pcd_timer(self)-> Tuple[str, str]:
        start_time = time.time()
        max_wait_time = int(int(os.getenv('DOWNLOAD_WAIT_TIME_MINS', '10'))*60)
        check_interval = 1
        while (time.time() - start_time) < max_wait_time:
            try:
                if self.frontend_Upload_completed():
                    docker_file_pth, ext = self.download_pointcloud()
                    if len(docker_file_pth) > 0:
                        return docker_file_pth, ext
            
            except Exception:
                pass  # Ignore errors and keep waiting
            # Wait before retry
            time.sleep(check_interval)
        # Timeout reached
        raise TimeoutError(f"Download not available after {max_wait_time} seconds")

    def frontend_Upload_completed(self)->bool:
        truth_map = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False
        }
        try:
            with TableClient.from_connection_string(conn_str=self.connection_string, table_name=self.root_log_table_name) as table_client:
                entity = table_client.get_entity(
                    partition_key=self.PartitionKey, 
                    row_key=self.row_key
                    )
                upload_completed_string = entity["upload_completed"]
                upload_completed = truth_map.get(upload_completed_string.lower())
                if upload_completed is None:
                    raise TypeError(f"Data From Data-Tables is Wrong, [{self.PartitionKey}, {self.row_key}]")
                return upload_completed
        except Exception as e:
            print(f"Exception occured in [frontend_Upload_completed] , \nError : [{e}]")
            return False
            
            

    def download_pointcloud(self)-> Tuple[str, str]:
        try:
            download_file_path = self.download_full_path
            docker_file_path = f"{self.docker_input_folder}/{self.filename}{self.download_file_extension}"
            print(f"\n\n\
                Download_file_path : {download_file_path}\n\
                docker_file_path : {docker_file_path}\n\
                    ")
            self.blob_obj.download_file(download_file_path, docker_file_path)
            return docker_file_path, self.download_file_extension
        except Exception as e:
            self.process_error(f"Error at Docker Download_pcd: at [ {download_file_path} ]\n[{e}]")
            return "",""
    
    def upload_everything(self, dir_path):
        try:
            
            for path, subdirs, files in os.walk(dir_path):
                for name in files:
                    fullPath=os.path.join(path, name)
                    file=fullPath.replace(dir_path,'')
                    fileName=file[1:len(file)]
                    print("FullPath : "+fullPath)
                    print("File Name :"+fileName)
                    print("\nUploading to Azure Storage as blob:\n\t" + fileName)
                    # self.blob_obj.blob_container_client.upload_blob(f"{self.process_folder}/{self.filename}/{fileName}", open(fullPath, "rb"))
                    self.blob_obj.blob_container_client.upload_blob(f"{self.process_folder}/{fileName}", open(fullPath, "rb"))
        except Exception as e:
            print(f"\n\n\nERROR at UPLOAD EVERYTHING : {e}\n\n\n")
    
    async def upload_everything_async(self, dir_path, tree_count:int):
        #TODO
        from azure.storage.blob.aio import BlobServiceClient
        try:
            blob_service = BlobServiceClient.from_connection_string(self.connection_string)
            async with blob_service.get_container_client(self.storageContainer) as blob_client:
                uploadingBlobs = []
                for path, subdirs, files in os.walk(dir_path):
                    for name in files:
                        print(name)
                        fullPath=os.path.join(path, name)
                        file=fullPath.replace(dir_path,'')
                        fileName=file[1:len(file)]
                        print(f"docker upload {self.process_folder}/{fileName}")
                        # uploadingBlobs.append(blob_client.upload_blob(f"{self.process_folder}/{self.filename}/{fileName}", open(fullPath, "rb"), overwrite=True))
                        uploadingBlobs.append(blob_client.upload_blob(f"{self.process_folder}{fileName}", open(fullPath, "rb"), overwrite=True))
            n_awaitables = len(uploadingBlobs)
            n_awaitables = len(uploadingBlobs)
            completed_count = 0
            for first_completed in asyncio.as_completed(uploadingBlobs):
                res = await first_completed
                completed_count+=1
                self.store_percentage((completed_count/n_awaitables)*100)
            
            self.store_percentage(100)
            await asyncio.sleep(1)
            self.store_completed(tree_count)
            await blob_service.close()
            return True
        except Exception as e:
            self.store_error(error_msg=f"Error at Storing {e}")
            return False
    
    def process_completed(self, coordinates:str, tree_count:int):
        if coordinates not in ["XYZ", "Lat/Long"]:
            raise ValueError(f"coordinates must be in ['XYZ','Lat/long'], your coordinates are [{coordinates}]")
        super().process_completed(tree_count) # Not a bug, if process is completed it should be store
        self.update_status(status="store", process_completed=str(True), coordinates=coordinates, trees_completed=tree_count)
    
    def store_completed(self, tree_count: int):
        if not isinstance(tree_count, int):
            raise TypeError(f"tree_count : [{tree_count}] must be type int, you have type [{type(tree_count)}]")
        super().store_completed(tree_count)
        self.update_status(status="completed", store_completed=str(True), trees_completed=tree_count)
    
    def store_error(self, error_msg:str):
        super().store_error(error_msg)
        self.update_status(error=str(True), error_msg=error_msg)
    
    def process_error(self, error_msg:str):
        super().process_error(error_msg)
        self.update_status(error=str(True), error_msg=error_msg)
        
    def update_status(self, status=None, process_completed=None, store_completed=None, error=None, error_msg=None, coordinates=None, trees_completed=None):
        try:
            with TableClient.from_connection_string(conn_str=self.connection_string, table_name=self.root_log_table_name) as table_client:
                entity = table_client.get_entity(partition_key=self.PartitionKey, row_key=self.row_key)
                if status is not None: # [upload, process, store, completed, replaced]
                    entity["status"] = status
                if process_completed is not None:
                    entity["process_completed"]=process_completed
                    entity["coordinates"] = coordinates
                    entity["trees_completed"] = trees_completed
                if store_completed is not None:
                    entity["store_completed"] = store_completed
                    entity["completed"] = store_completed
                    entity["trees_completed"] = trees_completed
                if error is not None and error_msg is not None:
                    entity["error"] = error
                    entity["error_msg"] = error_msg
                table_client.upsert_entity(mode=UpdateMode.MERGE, entity=entity)
                
                return True
        except Exception as e:
            print(f"Error occured in [update_state], {e}")
            return False
        return False
    
    def query_table_by_key_value(self, keys:list=["RowKey","ext"], values:list=["filename",".txt"]):
        try:
            results = []
            if len(keys) != len(values):
                assert f"Keys and Values have different length"
        
            with TableClient.from_connection_string(conn_str=self.connection_string, table_name=self.root_log_table_name) as table_client:
                query_string = "" # E.g. "key2 eq 'value1' and key2 eq 'value2'"
                for i, (key, value) in enumerate(zip(keys, values)):
                    if i==0:
                        query_string = query_string+f"{key} eq '{value}'"
                    else:
                        query_string = query_string+f" and {key} eq '{value}'"
                entities = table_client.query_entities(query_string)
                
                for entity in entities:
                    results.append(entity)
                    
            return results
        except Exception as e:
            print(f"Error occured in [query_table_by_key_value] {e}")
            return results