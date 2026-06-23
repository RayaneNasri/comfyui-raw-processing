from .read_raw_sensor_node import ReadRawSensorNode
from .read_raw_batch_node import BatchReadRawSensorNode

NODE_CLASS_MAPPINGS = {
    "ReadRawSensorNode": ReadRawSensorNode,
    "BatchReadRawSensorNode": BatchReadRawSensorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadRawSensorNode": "Read RAW Sensor (Single)",
    "ReadRawSensorBatchNode": "Read RAW Sensor (Batch/Burst)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
