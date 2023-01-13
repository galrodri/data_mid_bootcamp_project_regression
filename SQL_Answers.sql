-- Question 1, 2 and 3 have been resolved using Import Wizard

-- Question 4
SELECT * 
FROM house_price_data
ORDER BY id;

-- Question 5
ALTER TABLE house_price_data
DROP COLUMN date;

SELECT * 
FROM house_price_data
LIMIT 10;

-- Question 6
SELECT COUNT(*)
FROM house_price_data;

-- Question 7
SELECT DISTINCT bedrooms
FROM house_price_data
ORDER BY 1;

SELECT DISTINCT bathrooms
FROM house_price_data
ORDER BY 1;

SELECT DISTINCT floors
FROM house_price_data
ORDER BY 1;

SELECT DISTINCT a.condition
FROM house_price_data a
ORDER BY 1;

SELECT DISTINCT grade
FROM house_price_data
ORDER BY grade;

-- Question 8 
SELECT DISTINCT id
FROM house_price_data
ORDER BY price DESC
LIMIT 10;

-- Question 9
SELECT DISTINCT ROUND(AVG(price),2) as Average_House_Price
FROM house_price_data;

-- Question 10
SELECT DISTINCT bedrooms
, ROUND(AVG(price),2) as Average_House_Price
FROM house_price_data
GROUP BY 1
ORDER BY 1;

SELECT DISTINCT bedrooms
, AVG(sqft_living) as Average_Living_Size
FROM house_price_data
GROUP BY 1
ORDER BY 1;

SELECT waterfront
, ROUND(AVG(price),2) as Average_House_Price
FROM house_price_data
GROUP BY 1;

SELECT a.condition
, AVG(grade) as Average_Grade
FROM house_price_data a
GROUP BY 1
ORDER BY 1;

-- Question 11
SELECT *
FROM house_price_data a
WHERE bedrooms in (3,4)
AND bathrooms > 3 
AND floors = 1
AND waterfront = 0
AND a.condition >= 3
AND grade >= 5
AND price < 300000;

-- Question 12
SELECT * FROM house_price_data
WHERE price >= 2*(SELECT ROUND(AVG(price),2
					FROM house_price_data);

-- Question 13
DROP VIEW properties_double_avg_price;
CREATE VIEW properties_double_avg_price AS
			SELECT *
			FROM house_price_data
			WHERE price >= 2*(SELECT ROUND(AVG(price),2) 
								FROM house_price_data);

-- Question 14
SELECT B.Average_Price_4 - A.Average_Price_3 AS Average_Price_Delta
FROM (SELECT ROUND(AVG(price),2) AS Average_Price_4
			FROM house_price_data
			WHERE bedrooms = 4) B
CROSS JOIN
		(SELECT ROUND(AVG(price),2) AS Average_Price_3
		FROM house_price_data
		WHERE bedrooms = 3) A;

-- Question 15
SELECT DISTINCT zipcode
FROM house_price_data;

-- Question 16
SELECT *
FROM house_price_data
WHERE yr_renovated !=0;

-- Question 17 
SELECT *
FROM (SELECT *
		, RANK() OVER (ORDER BY price DESC) AS Ranking
		FROM house_price_data) a
WHERE Ranking = 11;