create database amazon
use amazon
select *from amazonsalesdata

/*Clean and preprocess the data (e.g., convert data types, handle missing values)*/ 
DELETE FROM amazonsalesdata WHERE Price IS NULL;
DELETE FROM amazonsalesdata WHERE Rating IS NULL;

/*Perform exploratory data analysis to gain insights into the data--*/
/*Which type of Brand as avg Rating*/
select top 1 Brand,ROUND(avg(Rating),2) as avg_rating
from amazonsalesdata
group by Brand
order by avg_rating desc

/*Which Brand has the most rating, */
select top 1 Brand, SUM(Rating) as Total_rating from amazonsalesdata
group by Brand
order by Total_rating asc

/*which Title contributes most of the rating*/
select Title, Brand, Price, ROUND(sum(Rating),2) as Total_rating
from amazonsalesdata
group by Title, Brand, Price 


/*Analyze the sales trends of each product category using SQL queries*/
select Brand, SUM(Price) as Total_Price 
from amazonsalesdata
group by Brand;


/*Understand the impact of pricing on sales*/
SELECT Brand, AVG(Price) AS avg_price, AVG(Rating) AS avg_rating
FROM amazonsalesdata
GROUP BY Brand
ORDER BY avg_rating DESC;


/*Identify the most Brand using SQL queries*/ 
select Brand, SUM(Price) as Total_Price
from amazonsalesdata
group by Brand
order by Total_Price Desc

/*Determine the most popular bands and price*/
SELECT TOP 5  Brand, Price FROM amazonsalesdata ORDER BY  Price DESC

/*Analyze the effect of ratings on sales*/
select Rating, SUM(Price) as Total_Price
from amazonsalesdata
group by Rating
order by Total_Price Desc










