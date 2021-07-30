import numpy as np
import xlrd
import math
from keras.models import load_model
import os
import keras.backend as K
import pandas as pd
from numpy.core._multiarray_umath import ndarray
import xlsxwriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ***************************** Harmonic creator ***************************
def con_RealCurrent(data):
    new_Data = []
    for i in range(len(data)):
        new_Data.append(data[i] * 0.0155)
    return new_Data


# FFT
def FFT_Data(dataSample):
    tpCount = len(dataSample)
    values = np.arange(int(tpCount / 2))
    timePeriod = tpCount / samplingFrequency
    frequencies = values / timePeriod

    fourierTransform = np.fft.fft(dataSample) / len(dataSample)  # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(dataSample) / 2))]  # Exclude sampling frequency
    return abs(fourierTransform)


# Calculate R.M.S. value
def rms_Value(signal, samples):
    sum_of_square = 0
    for num in range(samples):
        sum_of_square = sum_of_square + (signal[num] * signal[num])

    rms_val = math.sqrt(sum_of_square / samples)
    return rms_val


# Input Data
def data_input(File):
    currentArray = []
    harmonic_aray = File
    points = 1665

    # Current sample (10)
    currentSample_1 = []

    for x in range(points):
        currentArray.append(harmonic_aray[x] - 490)
    for i in range(1665):
        # Current Data
        currentSample_1.append(currentArray[i])
    # Callibration
    currentSample_1 = con_RealCurrent(currentSample_1)

    # FFT------------------------------------------------------------------------
    # Calculate current harmonics
    harmonicsSample_1 = FFT_Data(currentSample_1)

    averageHarmonics = []
    freq = []
    for num in range(len(harmonicsSample_1)):
        val = (harmonicsSample_1[num])
        averageHarmonics.append(val)
        f = num * 5
        freq.append(f)

    fund_freq = 10
    harmonics = []
    # Current harmonics error correction
    for number in range(22):
        har_val = fund_freq * number
        val = float(averageHarmonics[har_val])
        harmonics.append(round(val, 4))
    # Calculate average harmonic values

    return harmonics


# *************************** End of Harmonic creator **********************


# ************************** Begin of Text to Excel **********************
def text_to_excel(path):
    file = open(path, "r")
    lines = file.readlines()
    val1 = []  # Current value for line 1
    val2 = []

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    for line in lines[1:]:
        sline = line.split(':')

        if RepresentsInt(line[0]):
            val1.append(int(sline[0]))
            # val2temp = sline[1]
            # val2.append(int(val2temp[:3]))
    return val1


# ************************** End of Text to Excel ************************


# ************************** Begin of NN use **********************
def prediction(Data):
    test2: ndarray = np.array([Data])
    result = model.predict(test2)
    predicted = np.argmax(result, axis=1)
    return predicted


# ************************** End of NN use ************************


# ************************ Begin of Lable update *******************
def lable_update():
    # Lable list
    Lable_List_Location = r"E:\study materials\project final\Python files\zzzz__Test__zzzz\Main algoritham\List\List.xlsx"
    wb = xlrd.open_workbook(Lable_List_Location)
    sheet = wb.sheet_by_index(0)
    Len_lable = sheet.nrows
    Lable = []
    for x in range(Len_lable):
        Lable.append(str(sheet.row_values(x)))
    return Lable


# ************************ End of lable update *********************


def Middle_of_NN(harmonics):
    test1 = harmonics
    test2: ndarray = np.array([test1])

    get_layer = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[6].output])
    layer_output = get_layer([test2, 0])[0]
    numpy_array = np.array(layer_output)
    transpose = numpy_array.T
    layer_need1 = ndarray.tolist(transpose)
    return layer_need1


# ************************ Combination Detection *********************
def combination(harmonics):
    harmonic = harmonics
    data_file = Middle_of_NN(harmonics)
    data_file_integers = [x for l in data_file for x in l]
    # Error calculation
    error = [[None for i in range(10)] for j in range(no_of_comb)]
    total_error = []
    for x in range(no_of_comb):
        temp = 0
        for y in range(points - 1):
            error[x][y] = (data_file_integers[y] - arr[x][y]) ** 2
            temp = error[x][y] + temp
        total_error.append(math.sqrt(temp))

    min_error = min(total_error)
    min_index = total_error.index(min_error)
    if min_error < 20:
        return lable_initial[min_index]  # Correct prediction
    else:
        temp_err = 0
        for i in range(len(harmonic)):
            temp_err = temp_err + harmonic[i]
        if temp_err > 0.1:
            re_data = "error"  # if detections are not correct
            return re_data
        else:
            return "Zero"  # For no devices in the system


# ************************ End of Combination detection **************


# ************************** Begin of Flask use **********************

# ************************** End of Flask use **********************


def postive(List):
    pos_count, neg_count = 0, 0
    # iterating each number in list
    for num in List:
        # checking condition
        if num >= 0:
            pos_count += 1
        else:
            neg_count += 1
    if pos_count > neg_count:
        return True
    else:
        return False


# ********************* Combinations read ***************************
combination_path: str = r"E:\study materials\project final\Python files\zzzz__Test__zzzz\Main algoritham\Data\2_Devices_Middleof_nn"
# read combinations and save to array
for root, dirs, files in os.walk(combination_path):
    no_of_comb = len(files)
    j = 0
    arr = [[None for i in range(10)] for j in range(no_of_comb)]
    lable_initial = []
    for filename in files:
        if filename.endswith(".xlsx"):
            file_location = combination_path + "\\" + filename
            wb = xlrd.open_workbook(file_location)
            sheet = wb.sheet_by_index(0)
            points = sheet.nrows
            lable_initial.append(filename[:-5])
            for x in range(points):
                row_val_set = sheet.row_values(x)
                arr[j][x] = round(row_val_set[1], 4)
            j = j + 1


# ****************** End of combination read **************************


# ****************** NN update **************************
def newFolderCreate(lable):
    directory = str(lable)
    # Parent Directory path
    parent_dir: str = r"path of nn training data exsist\\"
    # Path
    path = os.path.join(parent_dir, directory)
    # Create the directory
    os.mkdir(path)


def dataToFolder(lable, data):
    path: str = r"path of nn training data exsist\\" + str(lable)
    length = 0
    for root, dirs, files in os.walk(path):
        if len(files) > 0:
            length = len(files)
    book = xlsxwriter.Workbook(path + "\\" + str(length + 1) + ".xlsx")
    worksheet = book.add_worksheet()
    row = 0
    for val in data:
        worksheet.write(row, 0, val)
        row += 1
    book.close()

    nnUpdate()


def nnUpdate():
    data = []
    folder_path = "Harmonic 2021_02_09"
    categories = os.listdir(folder_path)
    labels = [i for i in range(len(categories))]
    noOfDevises = int(len(labels))

    from keras.utils import np_utils
    labels_category = np_utils.to_categorical(labels)
    label_dict = {}

    for i in range(len(categories)):
        label_dict[categories[i]] = labels[i]

    for category in categories:
        data_path = os.path.join(folder_path, category)
        files = os.listdir(data_path)
        for file in files:
            file_path = os.path.join(data_path, file)
            datas = pd.read_excel(file_path)
            unarange_data = np.array(datas)
            unarange_data = unarange_data[:22, 1]
            data.append([unarange_data, label_dict[category]])
    data_set = []
    target = []

    for feature, label in data:
        data_set.append(feature)
        target.append(label)
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_target, test_target = train_test_split(data_set, target, test_size=0)
    from keras.utils import np_utils
    new_train_target = np_utils.to_categorical(train_target)
    train_data = np.array(train_data)
    test_data: ndarray = np.array(test_data)
    new_train_target = np.array(new_train_target)

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.optimizers import Adam, SGD
    input_data = Input(shape=(21,))
    encoded = Dense(21, activation='relu')(input_data)
    encoded1 = Dense(44, activation='linear')(encoded)
    encoded2 = Dense(44, activation='relu')(encoded1)
    encoded3 = Dense(22, activation='linear')(encoded2)
    encoded4 = Dense(15, activation='relu')(encoded3)
    encoded5 = Dense(10, activation='relu')(encoded4)
    encoded6 = Dense(noOfDevises, activation='softmax')(encoded5)
    detector = Model(input_data, encoded6)
    adam = Adam(lr=0.01)
    sgd = SGD(lr=0.05)
    detector.compile(optimizer=adam, loss='categorical_crossentropy')
    detector.fit(train_data, new_train_target, epochs=500)
    detector.save("ProgramNeural.h5")


# ****************** End NN update **************************

# ****************** NN retrain **************************

# ****************** End NN update **************************

# Variables -------------------------------------------------------------------
# How many time points are needed i,e., Sampling Frequency 3330 samples per seconds
samplingFrequency = 2000
# At what intervals time points are sampled
samplingInterval = 1 / samplingFrequency
# Begin time period of the signals
beginTime = 0
# End time period of the signals
endTime = 0.2
# Time points
time = np.arange(beginTime, endTime, samplingInterval)

# Time array 0.0s to 0.1s
T = []  # time period
ts = 0  # sampling time
for i in range(1665):
    ts = ts + 0.2 / 1665
    T.append(ts)
# End of variables ---------------------------------------------------------------
model = load_model('ProgramNeural.h5')
exit_command = True
system_harmonics = [0] * 21
device_list = []
nnn = [0] * 1
transfer = ""
wattage = 0
HarmonicsToUpdate = []


def final(dataArray):
    Harmonics_temp = data_input(dataArray)  # Getting harmonics
    Harmonics = Harmonics_temp[:-1]  # getting harmonics from 1st index
    No_of_device = int(nnn[0])
    if No_of_device == 0:
        value = prediction(Harmonics)
        for i in range(21):
            system_harmonics[i] = Harmonics[i]
        sign = True  # Adding devise
    else:
        Harmonic_array = np.subtract(Harmonics, system_harmonics)
        New_harmonics = Harmonic_array.tolist()
        sign = postive(New_harmonics)
        if not sign:
            New_harmonics = [abs(ele) for ele in New_harmonics]
        for i in range(21):
            system_harmonics[i] = New_harmonics[i]
        value = prediction(New_harmonics)

    int_value = value[0]  # predicted index of the device
    label = lable_update()  # get the label order
    Identifier = True
    Predict_device_temp = str(label[int_value])
    Predict_device = Predict_device_temp[2:-2]  # predicted device label
    if Predict_device == 'Other':
        if No_of_device == 0:
            comb_label = combination(Harmonics)
            if comb_label == "error":
                Identifier = False
            elif comb_label == "Zero":
                Identifier = True
                device_list.clear()
            else:
                nnn.clear()
                nnn.append(No_of_device + 2)
                device_list.clear()
                x = comb_label.split("_")
                for element in x:
                    device_list.append(element)
                Identifier = True
        else:
            comb_label = combination(New_harmonics)
            if comb_label == "error":
                Identifier = False
            else:
                x = comb_label.split("_")
                for element in x:
                    device_list.append(element)
                Identifier = True
                if sign:
                    nnn.clear()
                    nnn.append(No_of_device + 2)
                else:
                    nnn.clear()
                    nnn.append(No_of_device - 2)

    else:
        Identifier = True
        if sign:
            nnn.clear()
            nnn.append(No_of_device + 1)
            device_list.append(Predict_device)
        else:
            device_list.remove(Predict_device)
    print(device_list, "  Identifier = ", Identifier, " Sign = ", sign, " No of Devices = ", No_of_device)
    HarmonicsToUpdate = New_harmonics
    index()


addOrremove = 0
from flask import Flask, request
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    device_string = '.'.join(map(str, device_list))
    transmit = device_string + "@" + addOrremove + "@" + wattage
    return transmit


@app.route('/update', methods=['GET', 'POST'])
def index2():
    if (request.method == 'GET'):
        device_string = '.'.join(map(str, device_list))
        transmit = device_string + "@" + addOrremove + "@" + wattage
        return transmit
    elif (request.method == 'POST'):
        value = request.form['value']
        newFolderCreate(value)
        dataToFolder(value, HarmonicsToUpdate)


@socketio.on('connect')
def connect():
    print(request.sid)
    print("connecting.....................")


@socketio.on('samples')
def samples(message):
    print("Samples Arrived.....................")
    print(message)
    addOrremove = 1
    final(message)


if __name__ == '__main__':
    socketio.run(app, host='192.168.1.4', port=8089)
