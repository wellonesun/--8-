import json
import os
import time
import cv2
import numpy as np
import random
from abc import ABC, abstractmethod
import acl

# 常量定义
SUCCESS = 0
FAILED = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2
MEMCPY_DEVICE_TO_HOST = 1
MEMCPY_HOST_TO_DEVICE = 0

# 资源初始化函数
def init_acl(device_id):
    ret = acl.init()
    if ret != SUCCESS:
        raise RuntimeError(f"ACL init failed with error code: {ret}")
    
    ret = acl.rt.set_device(device_id)
    if ret != SUCCESS:
        raise RuntimeError(f"Set device failed with error code: {ret}")
    
    context, ret = acl.rt.create_context(device_id)
    if ret != SUCCESS:
        raise RuntimeError(f"Create context failed with error code: {ret}")
    
    print('ACL initialized successfully')
    return context

def deinit_acl(context, device_id):
    ret = acl.rt.destroy_context(context)
    if ret != SUCCESS:
        print(f"Destroy context failed with error code: {ret}")
    
    ret = acl.rt.reset_device(device_id)
    if ret != SUCCESS:
        print(f"Reset device failed with error code: {ret}")
    
    ret = acl.finalize()
    if ret != SUCCESS:
        print(f"ACL finalize failed with error code: {ret}")
    
    print('ACL deinitialized successfully')

# 模型基类
class Model(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.input_dataset = None
        self.output_dataset = None
        self.model_desc = None
        self._input_num = 0
        self._output_num = 0
        self._output_info = []
        self._is_released = False
        self._init_resource()
        self._create_input_dataset()
    
    def _init_resource(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != SUCCESS:
            raise RuntimeError(f"Load model failed with error code: {ret}")
        
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != SUCCESS:
            raise RuntimeError(f"Get model desc failed with error code: {ret}")
        
        self._input_num = acl.mdl.get_num_inputs(self.model_desc)
        self._output_num = acl.mdl.get_num_outputs(self.model_desc)
        
        self._gen_output_dataset()
        
        for i in range(self._output_num):
            dims = acl.mdl.get_output_dims(self.model_desc, i)[0]["dims"]
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)
            self._output_info.append({"shape": tuple(dims), "type": datatype})
        
        dims, ret = acl.mdl.get_input_dims_v2(self.model_desc, 0)
        if ret != SUCCESS:
            raise RuntimeError(f"Get input dims failed with error code: {ret}")
        dims = dims['dims']
        
        self.model_batch = dims[0]
        self.model_channel = dims[1]
        self.model_height = dims[2]
        self.model_width = dims[3]
        
        print(f"Model loaded: input shape {dims}, outputs: {self._output_num}")

    def _gen_output_dataset(self):
        self.output_dataset = acl.mdl.create_dataset()
        for i in range(self._output_num):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
            if ret != SUCCESS:
                raise RuntimeError(f"Malloc output buffer failed with error code: {ret}")
            
            data_buf = acl.create_data_buffer(buffer, size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buf)
            if ret != SUCCESS:
                raise RuntimeError(f"Add dataset buffer failed with error code: {ret}")
    
    def _create_input_dataset(self):
        self.input_dataset = acl.mdl.create_dataset()
        for i in range(self._input_num):
            input_size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(input_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            if ret != SUCCESS:
                raise RuntimeError(f"Malloc input buffer failed with error code: {ret}")
            
            data_buf = acl.create_data_buffer(buffer, input_size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buf)
            if ret != SUCCESS:
                raise RuntimeError(f"Add input dataset buffer failed with error code: {ret}")
    
    def execute(self, input_list):
        # 拷贝数据到输入缓冲区
        for i, input_data in enumerate(input_list):
            buf = acl.mdl.get_dataset_buffer(self.input_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buf)
            size = acl.get_data_buffer_size(buf)
            
            # 确保输入数据是连续的
            if not input_data.flags['C_CONTIGUOUS']:
                input_data = np.ascontiguousarray(input_data)
            
            # 直接拷贝到设备内存
            acl.rt.memcpy(
                data_ptr, 
                size, 
                acl.util.bytes_to_ptr(input_data.tobytes()),
                input_data.nbytes,
                MEMCPY_HOST_TO_DEVICE
            )
        
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != SUCCESS:
            error_msg = self._get_error_message(ret)
            raise RuntimeError(f"Model execute failed with error code: {ret} - {error_msg}")
        
        return self._get_output_data()
    
    def _get_error_message(self, error_code):
        # 常见的AscendCL错误代码
        error_messages = {
            500001: "ACL_ERROR_INVALID_PARAM",
            500002: "ACL_ERROR_INVALID_MODEL",
            500003: "ACL_ERROR_MODEL_SIZE_INVALID",
            500004: "ACL_ERROR_UNSUPPORTED_MODEL",
            500005: "ACL_ERROR_MODEL_EXECUTE",
            500006: "ACL_ERROR_INVALID_SHAPE",
            500007: "ACL_ERROR_INVALID_DATA",
            500008: "ACL_ERROR_MEMORY_ALLOCATION",
            500009: "ACL_ERROR_MEMORY_COPY",
            500010: "ACL_ERROR_DEVICE_NOT_FOUND"
        }
        return error_messages.get(error_code, "Unknown error")
    
    def _get_output_data(self):
        outputs = []
        for i in range(self._output_num):
            buf = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buf)
            size = acl.get_data_buffer_size(buf)
            
            np_type = self._get_numpy_type(self._output_info[i]["type"])
            shape = self._output_info[i]["shape"]
            
            # 创建空数组接收数据
            np_array = np.zeros(shape, dtype=np_type)
            
            # 直接从设备内存拷贝到numpy数组
            acl.rt.memcpy(
                np_array.ctypes.data,  # 目标地址
                np_array.nbytes,       # 目标大小
                data_ptr,              # 源地址
                size,                  # 源大小
                MEMCPY_DEVICE_TO_HOST
            )
            
            outputs.append(np_array)
        
        return outputs
    
    def _get_numpy_type(self, acl_type):
        type_map = {
            0: np.float32,   # ACL_FLOAT
            1: np.float16,   # ACL_FLOAT16
            3: np.int32,     # ACL_INT32
            8: np.uint32,    # ACL_UINT32
            12: np.bool_     # ACL_BOOL
        }
        return type_map.get(acl_type, np.float32)
    
    def release(self):
        if self._is_released:
            return
        
        if self.input_dataset:
            self._release_dataset(self.input_dataset)
        if self.output_dataset:
            self._release_dataset(self.output_dataset)
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
        
        self._is_released = True
        print("Model resources released")
    
    def _release_dataset(self, dataset):
        if not dataset:
            return
        
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            buf = acl.mdl.get_dataset_buffer(dataset, i)
            if buf:
                data_ptr = acl.get_data_buffer_addr(buf)
                if data_ptr:
                    acl.rt.free(data_ptr)
                acl.destroy_data_buffer(buf)
        
        acl.mdl.destroy_dataset(dataset)
    
    @abstractmethod
    def infer(self, inputs):
        pass

# YOLOv5模型类
class YOLOv5(Model):
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45, class_names=None):
        super().__init__(model_path)
        self.conf_threshold = conf_thresh
        self.iou_threshold = iou_thresh
        self.last_process_time = 0  # 记录最后一次处理时间
        
        if class_names is None:
            self.class_names = ["sea_cucumber", "scallop", "sea_urchin", "starfish"]
        else:
            self.class_names = class_names
        
        self.colors = {
            cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for cls in range(len(self.class_names))
        }
        
        # 时间统计
        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        self.frame_count = 0
    
    def init_dvpp(self, device_id):
        """DVPP初始化占位符"""
        print("DVPP acceleration not implemented in this version")
    
    def preprocess(self, image):
        """优化预处理：使用硬件加速（如果可用）"""
        start_time = time.time()
        
        orig_h, orig_w = image.shape[:2]
        target_width = self.model_width
        target_height = self.model_height
        
        # 直接缩放到模型尺寸
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 格式转换
        input_img = resized.astype(np.float32) * (1/255.0)
        input_img = input_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        input_img = np.ascontiguousarray(input_img, dtype=np.float32)
        
        # 保存缩放比例（用于后处理）
        self.scale_x = target_width / orig_w
        self.scale_y = target_height / orig_h
        
        self.preprocess_time += time.time() - start_time
        return input_img
    
    def infer(self, image):
        preprocessed = self.preprocess(image)
        
        infer_start = time.time()
        outputs = self.execute([preprocessed])
        self.inference_time += time.time() - infer_start
        
        post_start = time.time()
        results = self.postprocess(outputs, image.shape)
        self.postprocess_time += time.time() - post_start
        
        self.last_process_time = (time.time() - infer_start) * 1000  # ms
        self.frame_count += 1
        return results
    
    def postprocess(self, outputs, original_shape):
        orig_h, orig_w = original_shape[:2]
        pred = outputs[0][0]  # [num_boxes, 5+classes]
        
        detections = self.non_max_suppression(pred, self.conf_threshold, self.iou_threshold)
        
        results = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            class_id = int(det[5])
            
            # 应用缩放比例转换回原始坐标
            x1 /= self.scale_x
            y1 /= self.scale_y
            x2 /= self.scale_x
            y2 /= self.scale_y
            
            # 边界检查
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(orig_w-1, x2)), int(min(orig_h-1, y2))
            
            # 过滤太小或无效的检测
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
                
            results.append({
                "box": [x1, y1, x2, y2],
                "score": float(conf),
                "class_id": class_id,
                "class_name": self.class_names[class_id]
            })
        
        return results
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """优化：使用单次NMS代替按类别循环"""
        mask = prediction[..., 4] > conf_thres
        boxes = prediction[mask]
        
        if boxes.shape[0] == 0:
            return []
        
        # 获取类别信息
        class_conf = np.max(boxes[:, 5:], axis=1)
        class_pred = np.argmax(boxes[:, 5:], axis=1)
        scores = boxes[:, 4] * class_conf
        
        # 提取xywh格式 (模型原始输出)
        boxes_xywh = boxes[:, :4]
        
        # 转换为xyxy用于最终输出
        boxes_xyxy = self.xywh2xyxy(boxes_xywh)
        
        # 优化：一次性处理所有类别的NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(), 
            scores=scores.tolist(), 
            score_threshold=conf_thres, 
            nms_threshold=iou_thres
        )
        
        if indices is None or len(indices) == 0:
            return []
        
        detections = []
        indices = indices.flatten()
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            conf = scores[i]
            class_id = class_pred[i]
            detections.append([x1, y1, x2, y2, conf, class_id])
        
        return detections
    
    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
    
    def draw_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_id = det["class_id"]
            score = det["score"]
            
            color = self.colors[class_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{self.class_names[class_id]}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def print_timing_stats(self):
        if self.frame_count == 0:
            return
            
        avg_preprocess = self.preprocess_time / self.frame_count * 1000
        avg_inference = self.inference_time / self.frame_count * 1000
        avg_postprocess = self.postprocess_time / self.frame_count * 1000
        total_avg = (self.preprocess_time + self.inference_time + self.postprocess_time) / self.frame_count * 1000
        
        print("\n===== 处理时间统计 =====")
        print(f"平均预处理时间: {avg_preprocess:.2f}ms")
        print(f"平均推理时间: {avg_inference:.2f}ms")
        print(f"平均后处理时间: {avg_postprocess:.2f}ms")
        print(f"平均总处理时间: {total_avg:.2f}ms")
        print(f"理论最大FPS: {1000/total_avg:.2f}")

# 主函数 - 全面优化
def run_underwater_detection():
    params = {
        'model_path': 'models/model.om',
        'video_path': 'data/underwater8.mp4',
        'output_path': 'output8.1.mp4',
        'device_id': 0,
        'conf_thresh': 0.6,
        'iou_thresh': 0.4,
        'max_skip_frames': 5,  # 最大跳帧数
        # 已移除 output_quality 参数
    }
    
    context = init_acl(params['device_id'])
    detector = YOLOv5(params['model_path'], params['conf_thresh'], params['iou_thresh'])
    
    # 初始化DVPP
    detector.init_dvpp(params['device_id'])
    
    cap = cv2.VideoCapture(params['video_path'])
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {params['video_path']}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 使用更兼容的编码器列表
    codecs = ['mp4v', 'avc1', 'X264', 'MJPG']  # 移除 hvc1 并添加更兼容的选项
    fourcc = 0
    for codec in codecs:
        fourcc_test = cv2.VideoWriter_fourcc(*codec)
        if fourcc_test != -1:
            fourcc = fourcc_test
            print(f"Using codec: {codec}")
            break
    
    if fourcc == 0:
        print("Warning: No suitable codec found, using default")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 移除不支持的 quality 参数
    out = cv2.VideoWriter(
        params['output_path'], 
        fourcc, 
        fps, 
        (orig_width, orig_height)
    )
    
    # 添加写入器检查
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {params['output_path']}")
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    detection_results = []
    last_detections = []
    
    # 预热：处理前几帧不计算时间
    print("Warming up...")
    for _ in range(3):
        ret, frame = cap.read()
        if not ret: break
        detector.infer(frame)
    
    start_time = time.time()  # 重置开始时间
    frame_count = 0
    processed_count = 0
    
    print(f"Starting underwater object detection")
    print(f"Original video: {orig_width}x{orig_height}")
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"Max skip frames: {params['max_skip_frames']}")
    
    # 动态跳帧控制
    dynamic_skip = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 动态跳帧逻辑
            if dynamic_skip > 0:
                dynamic_skip -= 1
                detections = last_detections.copy()
                detection_results.append({
                    "frame": frame_count,
                    "detection": "skipped",
                    "used_last": True
                })
            else:
                # 处理当前帧
                detections = detector.infer(frame)
                processed_count += 1
                last_detections = detections.copy()
                
                for det in detections:
                    detection_results.append({
                        "frame": frame_count,
                        "detection": det,
                        "used_last": False
                    })
                
                # 更新跳帧策略
                if detector.last_process_time > 40:  # 超时增加跳帧
                    dynamic_skip = min(params['max_skip_frames'], dynamic_skip + 1)
                elif detector.last_process_time < 20 and dynamic_skip > 0:  # 快速处理减少跳帧
                    dynamic_skip = max(0, dynamic_skip - 1)
            
            # 绘制检测结果
            result_frame = detector.draw_detections(frame.copy(), detections)
            
            # 添加性能统计信息
            elapsed_time = time.time() - start_time
            processing_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
            actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            fps_text = f"Proc FPS: {processing_fps:.2f}, Actual: {actual_fps:.2f}"
            cv2.putText(result_frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            skip_text = f"Skip: {dynamic_skip}, Processed: {processed_count}"
            cv2.putText(result_frame, skip_text, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 写入视频
            out.write(result_frame)
            
            # 控制台日志
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}/{total_frames} - Processed: {processed_count} - Proc FPS: {processing_fps:.2f} - Actual FPS: {actual_fps:.2f} - Skip: {dynamic_skip}")
    except Exception as e:
        print(f"Error during processing: {e}")
        cap.release()
        out.release()
        detector.release()
        deinit_acl(context, params['device_id'])
        raise
    
    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0
    actual_fps = frame_count / total_time if total_time > 0 else 0
    
    # 打印详细时间统计
    detector.print_timing_stats()
    
    print("\nDetection completed")
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {processed_count}")
    print(f"Skipped frames: {frame_count - processed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS (processed): {avg_fps:.2f}")
    print(f"Actual FPS (output): {actual_fps:.2f}")
    
    with open('detection_results.json', 'w') as f:
        serializable_results = []
        for result in detection_results:
            if result["detection"] == "skipped":
                serializable_results.append({
                    "frame": result["frame"],
                    "detection": "skipped",
                    "used_last": result["used_last"]
                })
            else:
                serializable_results.append({
                    "frame": result["frame"],
                    "detection": {
                        "box": result["detection"]["box"],
                        "score": result["detection"]["score"],
                        "class_id": result["detection"]["class_id"],
                        "class_name": result["detection"]["class_name"]
                    },
                    "used_last": result["used_last"]
                })
        
        json.dump({
            "video_path": params['video_path'],
            "model_path": params['model_path'],
            "input_size": [detector.model_width, detector.model_height],
            "total_frames": frame_count,
            "processed_frames": processed_count,
            "max_skip_frames": params['max_skip_frames'],
            "processing_time": total_time,
            "avg_fps": avg_fps,
            "output_fps": actual_fps,
            "detections": serializable_results
        }, f, indent=2)
    
    cap.release()
    out.release()
    detector.release()
    deinit_acl(context, params['device_id'])
    
    print("Output saved to:", params['output_path'])
    print("Detection results saved to detection_results.json")
    
    return {
        "output_video": params['output_path'],
        "results_file": "detection_results1.json",
        "total_frames": frame_count,
        "processed_frames": processed_count,
        "processing_time": total_time,
        "avg_fps": avg_fps,
        "actual_fps": actual_fps
    }

if __name__ == '__main__':
    run_underwater_detection()
