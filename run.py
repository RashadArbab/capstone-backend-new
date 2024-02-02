from app import create_app
from dotenv import load_dotenv
import os

app = create_app()

if __name__ == '__main__':

	# print(os.environ.get("DATABASE_URL"))
    app.run(host='0.0.0.0', port=4000)
