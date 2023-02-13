from flask import Flask, flash, redirect, render_template, request, session
from werkzeug.utils import secure_filename
import oss2
import os
import time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'tests/'
RESULT_FOLDER = 'fakes/'
# 阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
auth = oss2.Auth('', '')
# yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
# 填写Bucket名称。
bucket = oss2.Bucket(auth, 'https://oss-cn-beijing.aliyuncs.com', 'pkuss')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload():
    # if 'f' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)

    images = []
    files = request.files.getlist("f")
    for file in files:
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        print(file.filename)
        if file and allowed_file(file.filename):
            filename = time.strftime("%Y-%m-%d%H-%M-%S", time.localtime(time.time())) + secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            # print('upload_image filename: ' + filename)
            bucket.put_object_from_file('hwseg/' + filename, upload_path)
            flash('Image successfully uploaded, wait a second...')

            result_filename = validate(upload_path, filename)
            session['_flashes'].clear()
            flash('finish!')
            images.append(result_filename)
            # return render_template('index.html', filename=result_filename)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html', images=images)


def validate(real_a_path, real_a_filename):
    fake_b_filename = real_a_filename
    opt.dataroot = real_a_path
    opt.dataset_mode = 'unaligned_single_img'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    assert dataset.__len__() == 1
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results(realA,fakeB,realB)
        im = util.tensor2im(visuals['fake_B'])
        im2 = util.tensor2im(visuals['real_A'])
        save_path = os.path.join(app.config['RESULT_FOLDER'], fake_b_filename)
        util.save_image(np.concatenate((im2, im), axis=1), save_path, aspect_ratio=1.0)
        bucket.put_object_from_file('hwseg/results/' + fake_b_filename, save_path)
    return fake_b_filename


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    # return redirect(url_for('static', filename='tests/' + filename), code=301)
    return redirect("https://pkuss.oss-cn-beijing.aliyuncs.com/hwseg/results/" + filename)


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    app.run(host='0.0.0.0', debug=True)
