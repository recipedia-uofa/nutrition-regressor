import csv
import webbrowser
from PIL import Image
from PIL.ImageQt import ImageQt
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap

import requests
from io import BytesIO
import pandas as pd
import argparse

INPUT_CSV = '6663_19999.csv'

CSV_COLUMNS = [
    'id',
    'url',
    'title',
    'img_url',
    'servings',
    'prep_time',
    'rating',
    'reviews',
    'made_it_count',
    'calories',
    'total_fat',
    'saturated_fat',
    'cholesterol',
    'sodium',
    'potassium',
    'total_carbohydrates',
    'dietary_fiber',
    'protein',
    'sugars',
    'vitamin_a',
    'vitamin_c',
    'calcium',
    'iron',
    'thiamin',
    'niacin',
    'vitamin_b6',
    'magnesium',
    'folate',
    # Everything above this index is a <contains> relationship
]


def parse_data(data_csv):
    with open(data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_ALL)

        col_len = len(CSV_COLUMNS)

        for row in reader:
            row_arr = [col.replace(r'"', r'\"') for col in row]
            # Accumulate ingredients into one array
            row_arr[col_len - 1] = row_arr[col_len - 1:]
            yield row_arr[:col_len]


def open_page(recipe_name):
    search_url = 'http://www.nutritionrank.com/search/apachesolr_search/{}?filters=ss_cck_field_food_type:Recipe'\
        .format(recipe_name)
    webbrowser.open(search_url)


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.pic = QLabel(self)

    def showRecipe(self, recipe_name, recipe_img):
        self.setWindowTitle(recipe_name)
        pix = QPixmap.fromImage(recipe_img)
        self.pic.setPixmap(pix)
        self.resize(pix.width(), pix.height())
        self.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to parse raw recipe htmls')
    parser.add_argument(
        '--start',
        help='start index',
        action='store',
        type=int,
        default=0
    )
    args = parser.parse_args()

    df = pd.DataFrame(data=parse_data(INPUT_CSV), columns=(CSV_COLUMNS))

    app = QApplication(sys.argv)
    ex = App()

    for index, row in df.iterrows():
        if index < args.start:
            continue

        response = requests.get(row['img_url'])
        img = Image.open(BytesIO(response.content))
        qim = ImageQt(img)
        ex.showRecipe(row['title'], qim)
        open_page(row['title'])

        print('Index: {}\nCalories: {}\tProtein: {}\tCarbs: {}\tFat: {}'.format(index,
              row['calories'], row['protein'], row['total_carbohydrates'], row['total_fat']))

        if input() == 'q':
            break

    print("Done")
    sys.exit(app.exec_())
