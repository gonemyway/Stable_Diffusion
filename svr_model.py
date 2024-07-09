from flask import Flask, request, render_template
from text2img import create_pipeline, text2image

# Khởi tạo Flask app
app = Flask(__name__)

# Định nghĩa tham số
image_path = "static"

# Khởi tạo pipeline
pipeline = create_pipeline()

@app.route("/", methods = ['GET', 'POST'])
def index():
    if request.method == "GET":
        # Trả về giao diện trang web
        return render_template("index.html")
    else:
        # Xứ lý việc người dùng upload prompt lên -> Sinh ảnh -> Trả về kết quả
        user_input = request.form["prompt"]
        print("Start gen...")

        # Đưa prompt qua pipeline
        im = text2image(user_input, pipeline)
        print("Finish gen!")

        im.save(image_path)

        return render_template("index.html", image_url = image_path)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6969, use_reloader=False)

