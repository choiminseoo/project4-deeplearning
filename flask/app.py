from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import logging
from ultralytics import YOLO
import re

# 설정 및 로깅
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
logging.basicConfig(level=logging.DEBUG)  # 디버깅 메시지를 출력

# YOLO 모델 로드
model = YOLO('best.pt')

# 레시피 데이터 로드
recipes = pd.read_csv('/Users/ming/Downloads/project4/recipes_modify.csv', encoding='utf-8')

def voca_ingredients(ingredients):
  vocab={
      'Bean sprouts':'콩나물',
      'Enoki Mushroom':'팽이버섯',
      'Sesame':'깻잎',
      'Shiitake mushrooms':'표고버섯',
      'apple':'사과',
      'bacon':'베이컨',
      'beef':'소고기',
      'bread':'빵',
      'cabbage':'양배추',
      'calamari':'오징어',
      'carrot':'당근',
      'cheese':'치즈',
      'chicken':'닭',
      'chives':'부추',
      'cucumber':'오이',
      'duck':'오리',
      'egg':'달걀',
      'egg plant':'가지',
      'garlic':'마늘',
      'green onion':'대파',
      'kimchi':'김치',
      'king oyster mushroom':'새송이버섯',
      'lettuce':'상추',
      'milk':'우유',
      'mung bean sprout':'숙주',
      'napa cabbage':'배추',
      'onion':'양파',
      'paprika':'파프리카',
      'pasta noodles':'파스타',
      'pear':'배',
      'pepper':'고추',
      'pork':'돼지고기',
      'potato':'감자',
      'quail egg':'메추리알',
      'radish':'무',
      'ramen':'라면',
      'rice cake':'떡',
      'shrimp':'새우',
      'spam':'스팸',
      'squash':'애호박',
      'sweet potato':'고구마',
      'tofu':'두부',
      'tuna can':'참치캔',
      'water parsley':'미나리'
      }
  ingre=[]
  for item in ingredients:
    ingre.append(vocab[item])
  return ingre

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    recommended_recipes = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image_url = url_for('static', filename='uploads/' + file.filename)

            try:
                detected_ingredients = detect_ingredients(file_path)
                recommended_recipes = recommend_cooks(detected_ingredients)
            except Exception as e:
                logging.error(f"Error in processing: {e}")
                recommended_recipes = []
            
            return render_template('index.html', recipes=recommended_recipes, image_url=image_url)
    return render_template('index.html', image_url=image_url)

def detect_ingredients(image_path):
    results = model(image_path)
    detected_ingredients = set()

    for result in results:
        if hasattr(result, 'names'):
            # YOLO 모델이 클래스 인덱스를 반환하는 경우 처리
            names = result.names
            for index in result.boxes.cls:
                ingredient = names[int(index)]
                detected_ingredients.add(ingredient)
        else:
            logging.error("Model result does not have 'names' attribute.")
    
    detected_ingredients_list = list(detected_ingredients)
    logging.debug(f"Detected ingredients list: {detected_ingredients_list}")
    
    # 라벨을 한국어로 변환
    return voca_ingredients(detected_ingredients_list)

def count_num_ingredients(ingredients):
  ingredients = re.sub(r'[a-zA-Z0-9]', '',ingredients)
  ingredients = ingredients.strip().split('|')
  ingredient=[]
  for i in ingredients:
    if i[:-1]!='|':
      ingredient.append(i)
  return len(ingredient)
  
def count_matching_ingredients(ingredients, target_ingredients):
    ingredients = ingredients.strip().split(' ')
    if ingredients[0] == 'nan':
        matching_count = 0
        gap = 1000
    else:
        matching_count = 0
        for item in target_ingredients:
            for ingre in ingredients:
                if item == ingre:
                    matching_count += 1
        gap = abs(len(ingredients) - matching_count)
    return matching_count, gap

def recommend_cooks(target_ingredients):
    recipes['매칭수'], recipes['차이'] = zip(*recipes['체크할 재료'].apply(lambda x: count_matching_ingredients(str(x), target_ingredients)))
    recipes['재료수']=recipes['재료'].apply(lambda x:count_num_ingredients(str(x)))
    recipes['전체 차이수']=recipes['재료수']-recipes['매칭수']
    filtered_df=recipes[(recipes['전체 차이수']<1) & (recipes['재료수']>1)]


    filtered_df = filtered_df.sort_values(by=['전체 차이수','매칭수', '조회수'], ascending=[True,False,  True]).head(3)
    print(filtered_df[['요리명','재료', '체크할 재료',  '매칭수',  '차이',  '재료수',  '전체 차이수']])

    return filtered_df['요리명'].tolist()

if __name__ == '__main__':
    app.run(debug=True)
