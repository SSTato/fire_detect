import argparse
import os
import platform
import sys
from pathlib import Path
import datetime
import time
import torch
import requests
import json
import setproctitle
import base64
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, within_time_list, time_sync


@smart_inference_mode()
def run(
        weights='weights/yolov5m.pt',  # model path or triton URL
        source='rtsp://admin:zjst@123@192.168.1.182//Streaming/Channels/1',  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        p_save_address='runs/p_save_address',  # Alarm image storage address
        a_save_address='runs/a_save_address',  # Alarm information storage address
        host="http://test.superton.cn",
        url_path="/server/superton-iot-server-zjf/export/push/alarm",
        interval_time=3.0,  # Detection frame interval time
        cycle_time=3.0,  # Cycle time of alarm
        in_time_list=[[0, 24], [0, 24], [0, 24], [0, 24], [0, 24], [0, 24], [0, 24]],
        specified_points=np.array([(0, 0), (1, 0), (1, 1),(0, 1)]),

):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    p_save_address = increment_path(Path(p_save_address), exist_ok=exist_ok)  # increment run
    a_save_address = increment_path(Path(a_save_address), exist_ok=exist_ok)  # increment run

    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, specified_points=specified_points)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    time_list = [time_sync(), time_sync()]  # 时间列表
    alarm_list = []

    for path, im, im0s, vid_cap, s in dataset:

        # 每小时重新发送告警失败信息
        if alarm_list and time.localtime().tm_min == 0 and time.localtime().tm_sec == 0:
            response = requests.post(host + url_path, json=alarm_list, headers={"Referer": host})
            if response.status_code == 200:
                alarm_list.clear()
                print("警报信息发送成功")
        if within_time_list(in_time_list):
            current_times = time_sync()
            if current_times - time_list[0] > (interval_time + 0.001):  # 每隔1s推理一次
                time_list[0] = current_times

                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if nosave:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        if current_times - time_list[1] > (cycle_time + 0.001):  # 每隔5s警报一次
                            time_list[1] = current_times
                            im1 = annotator.result() # save detect images
                            image_bytes = cv2.imencode('.jpg', im1)[1].tobytes()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                            p_save_path = p_save_address / datetime.datetime.now().strftime("%Y-%m/%d")
                            if not os.path.exists(p_save_path):
                                p_save_path.mkdir(parents=True, exist_ok=True)
                            pic_save_path = str(p_save_path / f'{now_time}.jpg')
                            cv2.imwrite(pic_save_path, im1)
                            a_save_path = a_save_address / datetime.datetime.now().strftime("%Y-%m")
                            if not os.path.exists(a_save_path):
                                a_save_path.mkdir(parents=True, exist_ok=True)
                            ala_save_path = str(a_save_path / f'{datetime.datetime.now().day}.json')

                            data = {"timestamp": now_time, "msg": "alarm", "flag": "200", "alarm_type": "fire",
                                    "save_type": "picture", "pic_addr": pic_save_path}
                            # 以追加模式打开文件，并将数据写入
                            with open(ala_save_path, "a") as file:
                                json.dump(data, file, indent=4, ensure_ascii=False)
                                file.write(",\n")  # 换行分隔每条数据
                            data['image_base64'] = image_base64
                            # response = requests.post(host + url_path, json=data, headers={"Referer": host})
                            # if response.status_code == 200:
                            #     print("警报信息发送成功")
                            # else:
                            #     alarm_list.append(data)
                            #     print("警报信息发送失败")


                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    new_process_name = sys.argv[1]  # 设置自定义的进程名称
    setproctitle.setproctitle(new_process_name)
    run(source=sys.argv[2], conf_thres=float(sys.argv[3]), host=sys.argv[4], url_path=sys.argv[5],
        cycle_time=float(sys.argv[6]), interval_time=float(sys.argv[7]), classes=eval(sys.argv[8]),
        in_time_list=eval(sys.argv[9]), specified_points=np.array(eval(sys.argv[10])))
    # run()
