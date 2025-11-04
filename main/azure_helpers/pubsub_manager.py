import os
import asyncio
import time
from azure.messaging.webpubsubclient import WebPubSubClient, WebPubSubClientCredential
from azure.messaging.webpubsubservice import WebPubSubServiceClient
from azure.messaging.webpubsubclient.models import (
    OnConnectedArgs,
    OnGroupDataMessageArgs,
    OnDisconnectedArgs,
    CallbackType,
    WebPubSubDataType,
)



class PubSubManager:
    def __init__(self):
        pass
        
    def init_pubsub(self):
        self.PUBSUBGROUPNAME=os.getenv("PUBSUBGROUPNAME")
        self.PUBSUBURL=os.getenv("PUBSUBURL")
        self.write_token_url = self.PUBSUBURL
        self.group_name = self.PUBSUBGROUPNAME
        self.client_write = WebPubSubClient(credential=WebPubSubClientCredential(client_access_url_provider=self.write_token_url))
        self._init_callbacks()
        self.client_write.open()
        self.client_write.join_group(self.PUBSUBGROUPNAME)
        
    def pub_dict(self, dict_msg):
        self.client_write.send_to_group(self.group_name, dict_msg, WebPubSubDataType.JSON, no_echo=False, ack=False)
    
    def pub_string(self, str_msg):
        self.client_write.send_to_group(self.group_name, str_msg, WebPubSubDataType.TEXT, no_echo=False, ack=False)

    def _init_callbacks(self):
        self.client_write.subscribe(CallbackType.CONNECTED, self._on_connected)
        self.client_write.subscribe(CallbackType.DISCONNECTED, self._on_disconnected)
        self.client_write.subscribe(CallbackType.GROUP_MESSAGE, self._on_group_message)
    
    def _on_connected(self, msg: OnConnectedArgs):
        print("======== connected ===========")
        print(f"Connection {msg.connection_id} is connected")


    def _on_disconnected(self, msg: OnDisconnectedArgs):
        print("========== disconnected =========")
        print(f"connection is disconnected: {msg.message}")


    def _on_group_message(self, msg: OnGroupDataMessageArgs):
        print("========== group message =========")
        if isinstance(msg.data, memoryview):
            print(f"Received message from {msg.group}: {bytes(msg.data).decode()}")
        else:
            print(f"Received message from {msg.group}: {msg.data}")
        
    def process_percentage(self, percentage:float):
        json_msg = {
            "status":"process",
            "percentage": percentage,
            "completed": False,
            "has_error": False,
            "error_message": "",
            "tree_count":0
        }
        self.pub_dict(json_msg)
    
    def process_completed(self):
        json_msg = {
            "status":"process",
            "percentage": 100,
            "completed": True,
            "has_error": False,
            "error_message": "",
            "tree_count": 0
        }
        self.pub_dict(json_msg)

    def process_error(self, error_message):
        json_msg = {
            "status":"process",
            "percentage": "",
            "completed": False,
            "has_error": True,
            "error_message": error_message,
            "tree_count":0
        }
        self.pub_dict(json_msg)
        
    def store_percentage(self, percentage:float):
        json_msg = {
            "status":"store",
            "percentage": percentage,
            "completed": False,
            "has_error": False,
            "error_message": "",
            "tree_count": 0
        }
        self.pub_dict(json_msg)
    
    def store_completed(self, tree_count:int):
        json_msg = {
            "status":"store",
            "percentage": 100,
            "completed": True,
            "has_error": False,
            "error_message": "",
            "tree_count": tree_count
        }
        self.pub_dict(json_msg)

    def store_error(self, error_message):
        json_msg = {
            "status":"store",
            "percentage": "",
            "completed": False,
            "has_error": True,
            "error_message": error_message,
            "tree_count": 0
        }
        self.pub_dict(json_msg)