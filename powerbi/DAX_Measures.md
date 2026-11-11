# DAX: tables & measures for the Netflix dashboard

Create a **Measures** table first (Home → Enter data → name it `_Measures`, then delete its
column after adding the measures). Paste each block via **Modeling → New measure**.

## 1. Date table (Modeling → New table)

```DAX
Date =
ADDCOLUMNS (
    CALENDAR ( DATE ( 2008, 1, 1 ), DATE ( 2021, 12, 31 ) ),
    "Year", YEAR ( [Date] ),
    "Month No", MONTH ( [Date] ),
    "Month", FORMAT ( [Date], "MMM" ),
    "Quarter", "Q" & QUARTER ( [Date] ),
    "Weekday", FORMAT ( [Date], "ddd" )
)
```
Mark it as a date table (Table tools → Mark as date table → `Date[Date]`) and sort
`Date[Month]` by `Date[Month No]`.

## 2. Core KPIs

```DAX
Total Titles = DISTINCTCOUNT ( Titles[show_id] )

Movies = CALCULATE ( [Total Titles], Titles[type] = "Movie" )

TV Shows = CALCULATE ( [Total Titles], Titles[type] = "TV Show" )

Movie Share % = DIVIDE ( [Movies], [Total Titles] )

Avg Movie Duration (min) = AVERAGE ( Titles[duration_min] )

Avg Seasons per Show = AVERAGE ( Titles[seasons] )

Countries = DISTINCTCOUNT ( Title_Countries[country] )

Genres = DISTINCTCOUNT ( Title_Genres[genre] )

Directors = DISTINCTCOUNT ( Title_Directors[director] )

Median Years to Netflix = MEDIAN ( Titles[years_to_netflix] )
```

## 3. Growth & time intelligence

```DAX
Titles Added LY =
CALCULATE ( [Total Titles], DATEADD ( 'Date'[Date], -1, YEAR ) )

Titles Added YoY % =
VAR Prev = [Titles Added LY]
RETURN IF ( NOT ISBLANK ( Prev ), DIVIDE ( [Total Titles] - Prev, Prev ) )

Cumulative Titles =
CALCULATE (
    [Total Titles],
    FILTER ( ALL ( 'Date'[Date] ), 'Date'[Date] <= MAX ( 'Date'[Date] ) )
)

Recent Content % =
DIVIDE ( CALCULATE ( [Total Titles], Titles[release_year] >= 2015 ), [Total Titles] )

Adult Content % =
DIVIDE ( CALCULATE ( [Total Titles], Titles[maturity_group] = "Adults" ), [Total Titles] )

Kids Content % =
DIVIDE ( CALCULATE ( [Total Titles], Titles[maturity_group] IN { "Kids", "Older Kids" } ), [Total Titles] )
```

## 4. Dynamic text for cards / titles

```DAX
Top Genre =
FIRSTNONBLANK (
    TOPN ( 1, VALUES ( Title_Genres[genre] ), [Total Titles] ),
    1
)

Top Country =
FIRSTNONBLANK (
    TOPN ( 1, FILTER ( VALUES ( Title_Countries[country] ), Title_Countries[country] <> "" ), [Total Titles] ),
    1
)

Dashboard Subtitle =
"Showing " & FORMAT ( [Total Titles], "#,0" ) & " titles · "
    & FORMAT ( [Movie Share %], "0%" ) & " movies · top genre: " & [Top Genre]
```

## 5. ML model page

```DAX
Cluster Titles = [Total Titles]

Cluster Share % =
DIVIDE ( [Total Titles], CALCULATE ( [Total Titles], ALL ( Titles[cluster_name] ) ) )

Hybrid Genre Precision =
CALCULATE ( MAX ( Model_Evaluation[Genre Precision@10] ),
            Model_Evaluation[Method] = "Hybrid weighted (ours)" )

Hybrid People Hit =
CALCULATE ( MAX ( Model_Evaluation[People Hit@10] ),
            Model_Evaluation[Method] = "Hybrid weighted (ours)" )
```

## 6. Calculated columns (optional)

```DAX
-- in Titles
Duration Bucket =
SWITCH (
    TRUE (),
    Titles[type] = "TV Show", "TV Show",
    Titles[duration_min] < 60, "< 1 h",
    Titles[duration_min] < 90, "1–1.5 h",
    Titles[duration_min] < 120, "1.5–2 h",
    Titles[duration_min] < 150, "2–2.5 h",
    "2.5 h +"
)

Era = IF ( Titles[release_year] >= 2015, "Streaming era (2015+)", "Pre-2015" )
```
