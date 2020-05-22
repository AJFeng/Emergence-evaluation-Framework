# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:36:45 2019

@author: aijing
"""

import tkinter as tk  # 使用Tkinter前需要先导入
import os
import tkinter.messagebox
import matlab.engine
import scipy.io as sio

eng = matlab.engine.start_matlab()
 
# 第1步，实例化object，建立窗口window
window = tk.Tk()

# 第2步，给窗口的可视化起名字
window.title('Cotton emergence mapping')
 
# 第3步，设定窗口的大小(长 * 宽)
window.geometry('700x300')  # 这里的乘是小x
 
# 第4步，在图形界面上设定标签
Question = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
Question.set('Do you have a model?')
l_Question = tk.Label(window, textvariable=Question, fg='Black', font=('Arial', 14), width=30, height=2)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l_Question.place(x=100, y=50, anchor='nw')


No_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
No_text.set('I need to train a model using my own data set')
l_No = tk.Label(window, textvariable=No_text, fg='Black', font=('Arial', 12), width=40, height=2)
l_No.place(x=180, y=115, anchor='nw')


Yes_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
Yes_text.set('I have the model (.pth file)')
l_Yes = tk.Label(window, textvariable=Yes_text, fg='Black', font=('Arial', 12), width=25, height=2)
l_Yes.place(x=177, y=155, anchor='nw')

# 定义一个函数功能（内容自己自由编写），供点击Button按键时调用，调用命令参数command=函数名

def train_start():
    window.destroy()
    
    global train_window
    train_window = tk.Tk()
    train_window.title('Start to train a model')
    train_window.geometry('700x300')  # 这里的乘是小x
    
    train_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    train_text.set('Do you have the labeling file?')
    l_train_text = tk.Label(train_window, textvariable=train_text, fg='Black', font=('Arial', 14), width=30, height=2)
    # 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
    l_train_text.place(x=100, y=50, anchor='nw')

    
    label_botton = tk.Button(train_window, text='No', font=('Arial', 12), width=13, height=1, command=labeling_start)
    label_botton.place(x=50, y=120, anchor='nw')

    train_model_botton = tk.Button(train_window, text='Yes', font=('Arial', 12), width=13, height=1, command=training_set)
    train_model_botton.place(x=50, y=170, anchor='nw')

    label_botton_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    label_botton_text.set('I want to label my data set now')
    l_label_botton = tk.Label(train_window, textvariable=label_botton_text, fg='Black', font=('Arial', 12), width=25, height=2)
    l_label_botton.place(x=220, y=115, anchor='nw')


    train_model_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    train_model_text.set('I have the labeling file')
    l_train_model = tk.Label(train_window, textvariable=train_model_text, fg='Black', font=('Arial', 12), width=17, height=2)
    l_train_model.place(x=220, y=170, anchor='nw')

    train_window.mainloop()
    
        
def test_start():
    window.destroy()
    
    global test_window
    test_window = tk.Tk()
    test_window.title('Start to mapping')
    test_window.geometry('700x300')  # 这里的乘是小x
    
    Test_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Test_text.set('Choose your process:')
    l_Test_text = tk.Label(test_window, textvariable=Test_text, fg='Black', font=('Arial', 14), width=30, height=2)
    # 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
    l_Test_text.place(x=100, y=50, anchor='nw')

    
    GPS_assignment_botton = tk.Button(test_window, text='GPS assignment', font=('Arial', 12), width=13, height=1, command=GPS_assignment_extractImage)
    GPS_assignment_botton.place(x=50, y=120, anchor='nw')

    mapping_botton = tk.Button(test_window, text='Mapping', font=('Arial', 12), width=13, height=1, command=mapping_set)
    mapping_botton.place(x=50, y=170, anchor='nw')

    GPS_assignment_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    GPS_assignment_text.set('Located the seedlings from individual image frames')
    l_GPS_assignment = tk.Label(test_window, textvariable=GPS_assignment_text, fg='Black', font=('Arial', 12), width=40, height=2)
    l_GPS_assignment.place(x=220, y=115, anchor='nw')


    mapping_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    mapping_text.set('Map the cotton emergece of the whole filed')
    l_mapping = tk.Label(test_window, textvariable=mapping_text, fg='Black', font=('Arial', 12), width=40, height=2)
    l_mapping.place(x=220, y=170, anchor='nw')

    
    

    test_window.mainloop()

def mapping():
    
    #global model_address
    model_address=model_file_path.get()
    #global img_address
    img_address=Img_folder_path.get()
    #global gps_address
    gps_address=GPS_file_path.get()
    
    str=('python CNN5_predict.py'+' '+img_address+' '+gps_address+' '+model_address)
    p=os.system(str)
    print(p) #0表示 success ， 1表示 fail
    
    if (p==0):
        tkinter.messagebox.showinfo(title='Finish!', message='Success! The result is written in seedling_cpu.csv')
    #test_window.destroy()
    
def mapping_set():
    
    test_window.destroy()
    
    global mapping_set_window
    mapping_set_window = tk.Tk()
    mapping_set_window.title('Start to mapping')
    mapping_set_window.geometry('700x400')  # 这里的乘是小x
    
    Img_folder = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder.set('Img_folder:')
    l_Img_folder = tk.Label(mapping_set_window, textvariable=Img_folder, fg='Black', font=('Arial', 12), width=10, height=2)
    l_Img_folder.place(x=30, y=20, anchor='nw')
    
    global Img_folder_path 
    Img_folder_path= tk.Entry(mapping_set_window,show = None,width=70)#显示成明文形式
    Img_folder_path.insert('end', 'E:/matlab/0531phantom/segment/')
    Img_folder_path.place(x=37, y=70, anchor='nw')
    
    GPS_file = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    GPS_file.set('GPS file:')
    l_GPS_file = tk.Label(mapping_set_window, textvariable=GPS_file, fg='Black', font=('Arial', 12), width=9, height=2)
    l_GPS_file.place(x=30, y=100, anchor='nw')

    global GPS_file_path 
    GPS_file_path= tk.Entry(mapping_set_window, show = None,width=70)#显示成明文形式
    GPS_file_path.insert('end', 'E:/matlab/0531phantom/gps.csv')
    GPS_file_path.place(x=37, y=150, anchor='nw')
    
    model_file = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    model_file.set('Model file:')
    l_model_file = tk.Label(mapping_set_window, textvariable=model_file, fg='Black', font=('Arial', 12), width=10, height=2)
    l_model_file.place(x=30, y=180, anchor='nw')

    global model_file_path 
    model_file_path= tk.Entry(mapping_set_window, show = None,width=70)#显示成明文形式
    model_file_path.insert('end', 'E:/spyder/resnet18_20_add_row4.pth')
    model_file_path.place(x=37, y=230, anchor='nw')
    
    test_next1 = tk.Button(mapping_set_window, text='Start to map!', font=('Arial', 12), width=15, height=1, command=mapping)
    test_next1.place(x=37, y=280, anchor='nw')
    
    mapping_set_window.mainloop()
    
def training_set():
    train_window.destroy()
    
    global train_setting_window
    train_setting_window = tk.Tk()
    train_setting_window.title('Start to mapping')
    train_setting_window.geometry('700x400')  # 这里的乘是小x
    
    Img_folder = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder.set('Img_folder:')
    l_Img_folder = tk.Label(train_setting_window, textvariable=Img_folder, fg='Black', font=('Arial', 12), width=10, height=2)
    l_Img_folder.place(x=30, y=20, anchor='nw')
    
    global Img_folder_path 
    Img_folder_path= tk.Entry(train_setting_window,show = None,width=70)#显示成明文形式
    Img_folder_path.insert('end', 'E:/spyder/meter/')
    Img_folder_path.place(x=37, y=70, anchor='nw')
    
    label_file = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    label_file.set('label_file:')
    l_label_file = tk.Label(train_setting_window, textvariable=label_file, fg='Black', font=('Arial', 12), width=10, height=2)
    l_label_file.place(x=30, y=100, anchor='nw')
    
    global label_file_path 
    label_file_path= tk.Entry(train_setting_window,show = None,width=70)#显示成明文形式
    label_file_path.insert('end', 'E:/spyder/labelsForMeter.csv')
    label_file_path.place(x=37, y=150, anchor='nw')
    
    dataloder = tk.Button(train_setting_window, text='Train!', font=('Arial', 12), width=15, height=1, command=dataloder_start)
    dataloder.place(x=37, y=200, anchor='nw')
    
    train_setting_window.mainloop()
    
def GPS_assignment_extractImage():
    test_window.destroy()
    
    global extractImage_window
    extractImage_window = tk.Tk()
    extractImage_window.title('Start to mapping')
    extractImage_window.geometry('700x400')  # 这里的乘是小x
    
    Img_folder = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder.set('Image_frame_folder:')
    l_Img_folder = tk.Label(extractImage_window, textvariable=Img_folder, fg='Black', font=('Arial', 12), width=18, height=2)
    l_Img_folder.place(x=30, y=20, anchor='nw')
    
    global Img_folder_path 
    Img_folder_path= tk.Entry(extractImage_window,show = None,width=70)#显示成明文形式
    Img_folder_path.insert('end', 'E:\\matlab\\0531phantom\\Phantom_4\\data3\\')
    Img_folder_path.place(x=37, y=70, anchor='nw')
    
    Img_folder_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder_text.set('Cropped_image_folder:')
    l_Img_folder_text = tk.Label(extractImage_window, textvariable=Img_folder_text, fg='Black', font=('Arial', 12), width=20, height=2)
    l_Img_folder_text.place(x=30, y=100, anchor='nw')
    
    global segment_file_path 
    segment_file_path= tk.Entry(extractImage_window,show = None,width=70)#显示成明文形式
    segment_file_path.insert('end', 'E:\\matlab\\0531phantom\\segment\\')
    segment_file_path.place(x=37, y=150, anchor='nw')
    
    extractImage_botton = tk.Button(extractImage_window, text='Assign!', font=('Arial', 12), width=15, height=1, command=extractImage_start)
    extractImage_botton.place(x=37, y=200, anchor='nw')
    
    extractImage_window.mainloop()
    
def extractImage_start():
    
    Img_folder_address=Img_folder_path.get()
    segment_file_address=segment_file_path.get()
    
    #data=sio.loadmat('E://a.mat')
    sio.savemat('Img_folder_address.mat', {'myFolder': Img_folder_address})
    sio.savemat('segment_file_address.mat', {'path': segment_file_address})
    eng.extractMeterInImages(nargout=0)
    
    tkinter.messagebox.showinfo(title='Finish!', message='Finish! Cropped images are in the segment folder!')

def labeling_start():
    train_window.destroy()
    
    global labeling_window
    labeling_window = tk.Tk()
    labeling_window.title('Start to train a model')
    labeling_window.geometry('700x300')  # 这里的乘是小x
    
    labeling_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    labeling_text.set('Choose your process:')
    l_labeling_text = tk.Label(labeling_window, textvariable=labeling_text, fg='Black', font=('Arial', 14), width=30, height=2)
    # 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
    l_labeling_text.place(x=100, y=50, anchor='nw')

    
    segmetation_botton = tk.Button(labeling_window, text='Seedlings segmentation', font=('Arial', 12), width=20, height=1, command=segmetation_start)
    segmetation_botton.place(x=20, y=120, anchor='nw')

    crop_botton = tk.Button(labeling_window, text='Crop images', font=('Arial', 12), width=20, height=1, command=crop_start)
    crop_botton.place(x=20, y=170, anchor='nw')

    segmetation_botton_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    segmetation_botton_text.set('Segment the seedling clusters')
    l_segmetation_botton = tk.Label(labeling_window, textvariable=segmetation_botton_text, fg='Black', font=('Arial', 12), width=35, height=2)
    l_segmetation_botton.place(x=250, y=115, anchor='nw')


    crop_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    crop_text.set('Crop the images based on my manual labeling')
    l_crop = tk.Label(labeling_window, textvariable=crop_text, fg='Black', font=('Arial', 12), width=40, height=2)
    l_crop.place(x=250, y=170, anchor='nw')

    labeling_window.mainloop()

def segmetation_start():
    labeling_window.destroy()
    
    global segmetation_window
    segmetation_window = tk.Tk()
    segmetation_window.title('Start to train a model')
    segmetation_window.geometry('700x400')  # 这里的乘是小x
    
    Img_folder = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder.set('Img_folder:')
    l_Img_folder = tk.Label(segmetation_window, textvariable=Img_folder, fg='Black', font=('Arial', 12), width=10, height=2)
    l_Img_folder.place(x=30, y=20, anchor='nw')
    
    global Img_folder_path 
    Img_folder_path= tk.Entry(segmetation_window,show = None,width=70)#显示成明文形式
    Img_folder_path.insert('end', 'E:\\spyder\\row_4\\')
    Img_folder_path.place(x=37, y=70, anchor='nw')
    
    Output_file_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Output_file_text.set('Output_folder:')
    l_Output_file = tk.Label(segmetation_window, textvariable=Output_file_text, fg='Black', font=('Arial', 12), width=13, height=2)
    l_Output_file.place(x=30, y=100, anchor='nw')
    
    global Output_path 
    Output_path= tk.Entry(segmetation_window,show = None,width=70)#显示成明文形式
    Output_path.insert('end', 'E:\\spyder\\row_4\\outputs\\')
    Output_path.place(x=37, y=150, anchor='nw')
    
    segmetation_next1 = tk.Button(segmetation_window, text='Segment!', font=('Arial', 12), width=15, height=1, command=segmetation_next)
    segmetation_next1.place(x=37, y=200, anchor='nw')
    
    segmetation_window.mainloop()

def segmetation_next():
    Img_folder_address=Img_folder_path.get()
    Output_path_address=Output_path.get()
    
    #data=sio.loadmat('E://a.mat')
    sio.savemat('Img_folder_address.mat', {'img_path': Img_folder_address})
    sio.savemat('output_file_address.mat', {'output_path': Output_path_address})
    eng.GetLabelimage(nargout=0)
    
    tkinter.messagebox.showinfo(title='Finish!', message='Finish! Please visual label every seedling clusters!')

def crop_start():
    labeling_window.destroy()
    
    global crop_window
    crop_window = tk.Tk()
    crop_window.title('Start to train a model')
    crop_window.geometry('700x400')  # 这里的乘是小x
    
    Img_folder = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    Img_folder.set('The rotated image folder:')
    l_Img_folder = tk.Label(crop_window, textvariable=Img_folder, fg='Black', font=('Arial', 12), width=20, height=2)
    l_Img_folder.place(x=30, y=20, anchor='nw')
    
    global Img_folder_path 
    Img_folder_path= tk.Entry(crop_window,show = None,width=70)#显示成明文形式
    Img_folder_path.insert('end', 'E:\\matlab\\row4\\row4\\allimage\\')
    Img_folder_path.place(x=37, y=70, anchor='nw')
    
    labeling_file = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    labeling_file.set('Labeling file:')
    l_labeling_file = tk.Label(crop_window, textvariable=labeling_file, fg='Black', font=('Arial', 12), width=10, height=2)
    l_labeling_file.place(x=30, y=100, anchor='nw')

    global count_file_path 
    count_file_path= tk.Entry(crop_window, show = None,width=70)#显示成明文形式
    count_file_path.insert('end', 'E:\\matlab\\row4\\row4\\row4raw_count.csv')
    count_file_path.place(x=37, y=150, anchor='nw')
    
    count_file_text = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    count_file_text.set('Row_space file:')
    l_model_file = tk.Label(crop_window, textvariable=count_file_text, fg='Black', font=('Arial', 12), width=12, height=2)
    l_model_file.place(x=30, y=180, anchor='nw')

    global rowSpace_file_path
    rowSpace_file_path= tk.Entry(crop_window, show = None,width=70)#显示成明文形式
    rowSpace_file_path.insert('end', 'E:\\matlab\\row4\\row4\\row4_rowSpace.csv')
    rowSpace_file_path.place(x=37, y=230, anchor='nw')
    
    crop_next1 = tk.Button(crop_window, text='Start to crop!', font=('Arial', 12), width=15, height=1, command=crop_next)
    crop_next1.place(x=37, y=280, anchor='nw')
    
    crop_window.mainloop()
    
    
def crop_next():
    Img_folder_address=Img_folder_path.get()
    label_file_address=count_file_path.get()
    rowSpace_file_address=rowSpace_file_path.get()
    
    #data=sio.loadmat('E://a.mat')
    sio.savemat('Img_folder_address.mat', {'img_path': Img_folder_address})
    sio.savemat('label_file_address.mat', {'label_path': label_file_address})
    sio.savemat('rowSpace_file_address.mat', {'rowSpace_path': rowSpace_file_address})
    eng.imageGenerateByMeter(nargout=0)
    
    tkinter.messagebox.showinfo(title='Finish!', message='Finish! This is time to train your own model!')
    
def dataloder_start():
    
    #global model_address
    label_file=label_file_path.get()
    #global img_address
    img_address=Img_folder_path.get()
    
    
    str=('python CNN_4_pretrained_regression.py'+' '+label_file+' '+img_address)
    p=os.system(str)
    print(p) #0表示 success ， 1表示 fail
    
    if (p==0):
        tkinter.messagebox.showinfo(title='Finish!', message='Success! The model has been saved as resnet18.pth')
    #test_window.destroy()
 
# 第5步，在窗口界面设置放置Button按键
train_botton = tk.Button(window, text='No', font=('Arial', 12), width=10, height=1, command=train_start)
train_botton.place(x=50, y=120, anchor='nw')

test_botton = tk.Button(window, text='Yes', font=('Arial', 12), width=10, height=1, command=test_start)
test_botton.place(x=50, y=170, anchor='nw')

# 第6步，主窗口循环显示
window.mainloop()