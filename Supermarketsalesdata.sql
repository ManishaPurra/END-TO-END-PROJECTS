--Retrieve the sales data--
select *from supermarketsalesdata

--Clean and preprocess the data (e.g., convert data types, handle missing values)--
--convert data types--
SELECT CAST(Date AS DATE) as T_date from supermarketsalesdata;

--handle missing values--
SELECT * FROM supermarketsalesdata WHERE Total IS NOT NULL;

--Perform exploratory data analysis to gain insights into the data--
--Which type of product is the least sold?--
select top 1 Product_line, SUM(QUANTITY) as Total_quantity_sold from supermarketsalesdata
group by Product_line
order by Total_quantity_sold asc

--Which product line has the most rating, which is rated the least?--
select top 1 Product_line,ROUND(avg(Rating),2) as avg_rating
from supermarketsalesdata
group by Product_line
order by avg_rating desc

--Which gender contributes most to the sales?--
select Branch, City, Gender, ROUND(sum(Total),2) as Total_cost
from supermarketsalesdata
group by Branch, City, Gender


--Analyze the sales trends of each product category using SQL queries --
SELECT Product_line, SUM(Total) AS Total_sales
FROM supermarketsalesdata
GROUP BY Product_line;

--Determine the most popular products and product categories--
SELECT top 1 Product_line, SUM(Total) AS Total_sales
FROM supermarketsalesdata
GROUP BY Product_line
ORDER BY Total_sales Desc

--Understand the impact of pricing on sales--
SELECT FLOOR(Unit_price/10)*10 AS Price_range, SUM(Total) AS Total_sales
FROM supermarketsalesdata
GROUP BY FLOOR(Unit_price/10)*10;



--Identify the most valuable customers using SQL queries-- 
SELECT Customer_type, SUM(Total) AS Total_sales
FROM supermarketsalesdata
GROUP BY Customer_type
ORDER BY Total_sales DESC;


--Analyze the effect of customer ratings on sales--
SELECT Rating, SUM(Total) AS Total_sales
FROM supermarketsalesdata
GROUP BY Rating;





