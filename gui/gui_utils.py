import subprocess


def detection_Popen(weights, filePath):
    command = r'python icsi.py splash --weights={} --video={}'.format(weights, filePath)
    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True, cwd=r'D:\MASK-RCNN\samples\icsi')
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code


def train_Popen(dataset, weights, epochs, steps, imGPU, layers):
    command = r'python icsi.py train --dataset={} --weights={} --epochs={} --steps={} --imGPU={} --layers={}' \
        .format(dataset, weights, epochs, steps, imGPU, layers)
    p = subprocess.Popen(["start", "cmd", "/k", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     shell=True, cwd=r'D:\MASK-RCNN\samples\icsi')
    ret_code = p.wait()
    print("ret_code ", ret_code)
    if ret_code != 0:
        print("Something went wrong...")
    return ret_code

