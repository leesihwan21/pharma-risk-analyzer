from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')
