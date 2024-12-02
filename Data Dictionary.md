### Data Dictionary
  
  # Walmart Sales Data

| Field | Type | Description |
|----------|----------|----------|
| Date | datetime | Date of sales record |
| Store | int | Store number identifier |
| Weekly_Sales | float | Total sales for the week |
| Holiday_Flag | int | 1 if holiday week, 0 otherwise |
| Temperature | float | Average Temperature in the region |
| Fuel_Price | float | Cost of fuel in the region |
| CPI | float | Consumer Price Index |
| Unemployment | float | Unemployement rate |

  # Amazon Sales Data

| Field | Type | Description |
|----------|----------|----------|
| order_date | datetime | Date of order placement |
| order_id | string | Unique order identifier |
| product_name | string | Name of the product |
| category | string | Product category hierarchy |
| qty_ordered | int | Quantity ordered |
| discounted_price | float | Final price after discount |
| actual_price | float | Original Price |
| discount_percentage | float | Percentage of discount applied |
| Region | string | Geographic region of sale |
| payment_method | string | Method of payment used |

  # Amazon Reviews Data

| Field | Type | Description |
|----------|----------|----------|
| product_name | string | Name of product |
| review_id | string | Unique review identifier |
| review_title | string | Title of the review |
| review_content | text | Main review content |
| rating | float | Product rating (1-5) |
| user_id | string | Unique user identifier |
| user_name | string | Username of reviewer |
| rating_count | int | Total Number of ratings |
