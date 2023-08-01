import sys
import subprocess
import psutil
import re

def check_string_form(pid_name):
    alphanumeric_pattern = r'^[a-zA-Z0-9]+\s*$'
    if re.match(alphanumeric_pattern, pid_name):
        if any(char.isalpha() for char in pid_name) and any(char.isdigit() for char in pid_name):
            return 0
            # "Alphanumeric (letters and numbers)"
        elif any(char.isalpha() for char in pid_name):
            return 1
            # "Alpha (letters only)"
        elif any(char.isdigit() for char in pid_name):
            return 2
            # "Numeric (numbers only)"
    return "Unknown form"


def start_b(python_name):
    print("开始启动", python_name)
    command = f'source activate pytorch && python {python_name} .py {" ".join(sys.argv[2:])}'
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def stop_b(process_name):
    print("开始停止", process_name)
    if check_string_form(process_name) in [1, 2]:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = ' '.join(proc.info['cmdline'])
            if process_name in cmdline:
                pid = proc.info['pid']
                try:
                    process_obj = psutil.Process(pid)
                    process_obj.terminate()  # 终止进程
                    print(f"进程 {pid}-{process_name} 已成功关闭")
                except psutil.NoSuchProcess:
                    print(f"进程 {pid}-{process_name} 不存在或已关闭")
                except psutil.AccessDenied:
                    print(f"无权限关闭进程 {pid}-{process_name}")
                except Exception as e:
                    print(f"关闭进程 {pid}-{process_name} 出现异常：{e}")
    elif check_string_form(process_name) == 0:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = ' '.join(proc.info['cmdline'])
            if process_name == cmdline.strip():
                pid = proc.info['pid']
                try:
                    process_obj = psutil.Process(pid)
                    process_obj.terminate()  # 终止进程
                    print(f"进程 {pid}-{process_name} 已成功关闭")
                except psutil.NoSuchProcess:
                    print(f"进程 {pid}-{process_name}不存在或已关闭")
                except psutil.AccessDenied:
                    print(f"无权限关闭进程 {pid}-{process_name}")
                except Exception as e:
                    print(f"关闭进程 {pid}-{process_name} 出现异常：{e}")
    return False



if __name__ == "__main__":

    if sys.argv[0] == '0':
        python_name = sys.argv[2]+sys.argv[1]+'.py'
        process_name = sys.argv[3]
        start_b(python_name)
    elif sys.argv[0] == '1':
        process_name = sys.argv[1]
        if stop_b(process_name):
            print("成功停止", process_name)
        else:
            print("未找到运行中的", process_name, "进程")



'''
开始算法：调用star_stop.py,传递参数   '0'、'算法名'、'算法路径'、'进程名'、'视频流地址'、'置信度'、'host'、'url_path'、'预警周期'、'取帧间隔'、'识别场景'、'运行时间'、'检测区域'、
停止算法：调用star_stop.py,传递参数   '1'、'进程名'
'''

