## 컨텐츠 기반 필터링을 이용한 영화 추천

- TMDB API를 이용해 영화 데이터를 받아온다.
- overview(줄거리)를 BOW피처 백터화
- cosine 유사도를 이용해 TF-IDF 행렬 생성
- 영화제목을 입력하면 입력된 영화와 overview가 유사한 영화 상위 10개 추천

<img src="https://user-images.githubusercontent.com/107156650/228000735-e4980996-55e1-4d32-9693-c19de5bb5f82.gif">