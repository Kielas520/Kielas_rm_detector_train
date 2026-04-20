import os
import sys
import cv2
import numpy as np
from ctypes import *
from pathlib import Path

# ================================================= #
# DLL 与 路径初始化
# ================================================= #
BASE_DIR = Path(__file__).resolve().parent
LIB_DIR = BASE_DIR.parent / "hik_lib"
MV_IMPORT_PATH = BASE_DIR.parent / "MvImport"

if LIB_DIR.exists() and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(str(LIB_DIR))
os.environ['PATH'] = str(LIB_DIR) + os.pathsep + os.environ['PATH']
sys.path.append(str(MV_IMPORT_PATH))

try:
    from MvCameraControl_class import *
    from CameraParams_const import *
except ImportError as e:
    print(f"❌ 导入 MvImport 失败: {e}")
    sys.exit()

class HikCamera:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cam = MvCamera()
        self.handle_created = False
        self.is_running = False
        self.stOutFrame = MV_FRAME_OUT()
        
        # 枚举设备
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, self.deviceList)
        
        if ret != 0 or self.deviceList.nDeviceNum <= device_index:
            self.opened = False
            print("❌ 未找到设备")
        else:
            self.opened = True
            self.stDeviceImg = cast(self.deviceList.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents

    def open(self):
        """打开相机并开始取流"""
        if not self.opened: return False
        
        # 创建句柄
        ret = self.cam.MV_CC_CreateHandle(self.stDeviceImg)
        if ret != 0: return False
        self.handle_created = True

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0: return False
        # --- 新增：解除帧率限制封印 ---
        # 关闭帧率限制，让相机能跑多快跑多快
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)
        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0: return False
        
        self.is_running = True
        return True

    def set_exposure(self, exposure_time, auto_mode=False):
        """
        设置曝光参数
        :param exposure_time: 曝光时间 (单位: us)
        :param auto_mode: 是否开启自动曝光
        """
        if not self.handle_created: return False

        if auto_mode:
            # 自动曝光模式: Continuous = 2
            ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        else:
            # 必须先关闭自动曝光 (Off = 0) 才能手动设置值
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
        
        return ret == 0

    def get_exposure(self):
        """获取当前曝光值 (us)"""
        if not self.handle_created: return 0.0
        stFloatValue = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatValue)
        return stFloatValue.fCurValue if ret == 0 else 0.0

    def read(self):
        """读取一帧，返回 (ret, frame)"""
        if not self.is_running:
            return False, None

        # 超时时间设为 1000ms
        ret = self.cam.MV_CC_GetImageBuffer(self.stOutFrame, 1000)
        if ret == 0:
            # 转换原始数据为 numpy 数组
            pData = (c_ubyte * self.stOutFrame.stFrameInfo.nFrameLen).from_address(
                cast(self.stOutFrame.pBufAddr, c_void_p).value
            )
            img_raw = np.frombuffer(pData, count=self.stOutFrame.stFrameInfo.nFrameLen, dtype=np.uint8)
            
            h, w = self.stOutFrame.stFrameInfo.nHeight, self.stOutFrame.stFrameInfo.nWidth
            
            # 根据像素格式处理图像
            # 这里的逻辑假设为 8-bit 数据
            image = img_raw.reshape((h, w))

            # 常见的 Bayer 格式转换 (根据相机实际型号可能需要调整 COLOR_BayerXX2BGR)
            # 如果是黑白相机，cvtColor 会抛出异常，直接返回原图即可
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
            except Exception:
                pass 

            self.cam.MV_CC_FreeImageBuffer(self.stOutFrame)
            return True, image
        else:
            return False, None

    def close(self):
        """释放资源"""
        if self.is_running:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.is_running = False
        if self.handle_created:
            self.cam.MV_CC_DestroyHandle()
            self.handle_created = False

    def __del__(self):
        self.close()