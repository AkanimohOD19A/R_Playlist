## Load Dependencies
# ADDITIONAL FIX: Add this helper function at the top of your script
`%||%` <- function(x, y) if(is.null(x) || length(x) == 0) y else x

required_packages <-c("rpart", "ranger", "broom", "baguette", "glmnet",
                      "tidymodels", "janitor", "scales", "xgboost",
                      "shinydashboard", "DT", "plotly", "baguette", "ranger")
missing_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if(length(missing_packages) > 0){
  install.packages(missing_packages)
}

suppressPackageStartupMessages({
  library(shiny)
  library(shinydashboard)
  library(DT)
  library(plotly)
  library(tidyverse)
  library(tidymodels)
  library(broom)
  library(scales)
  library(janitor)
  library(rpart)
  library(glmnet)
  library(xgboost)
  library(baguette)
  library(ranger)
})

setwd("C:/Users/AfrologicInsect/Documents/R-Playlist/TBM/playground-series-s5e4")

load_ensemble_model <- function(model_path = "C:/Users/AfrologicInsect/Documents/R-Playlist/TBM/playground-series-s5e4/TBM_ensemble_model.rds") {
  if (!file.exists(model_path)) {
    # Try alternative paths
    alt_paths <- c(
      "TBM_ensemble_model.rds",  # Current directory
      file.path(getwd(), "TBM_ensemble_model.rds"),  # Explicit current directory
      "C:/Users/AfrologicInsect/Documents/R-Playlist/TBM/playground-series-s5e4/TBM_ensemble_model.rds"  # Full path
    )
    
    # Check which path exists
    existing_path <- NULL
    for(path in alt_paths) {
      if(file.exists(path)) {
        existing_path <- path
        break
      }
    }
    
    if(is.null(existing_path)) {
      stop(paste("Model file not found in any of these locations:\n",
                 paste(alt_paths, collapse = "\n")))
    } else {
      model_path <- existing_path
    }
  }
  readRDS(model_path)
}

#Shiny optimized: Prediction function
create_shiny_predictor <- function(ensemble){
  function(input_data){
    tryCatch({
      if(is.null(input_data) || nrow(input_data) == 0){
        return(list(success = FALSE, error = "No input data provided"))
      }
      
      # Check for expected features
      missing_features <- setdiff(ensemble$expected_features, names(input_data))
      if(length(missing_features) > 0) {
        return(list(
          success = FALSE,
          error = paste("Missing features:", paste(missing_features, collapse = ", "))
        ))
      }
      # Generate base model predictions
      rf_pred <- predict(ensemble$base_models$rf_model, new_data = input_data)$.pred
      xgb_pred <- predict(ensemble$base_models$xgb_model, new_data = input_data)$.pred
      bag_pred <- predict(ensemble$base_models$bag_model, new_data = input_data)$.pred
      
      meta_features <- data.frame(
        rf_pred = rf_pred,
        xgb_pred = xgb_pred,
        bag_pred = bag_pred
      )
      
      # Final ensemble prediction
      final_pred <- predict(ensemble$meta_model, new_data = meta_features)$.pred
      
      # Return detailed results
      return(list(
        success = TRUE,
        ensemble_prediction = round(final_pred, 2),
        individual_predictions = list(
          random_forest = round(rf_pred, 2),
          xgboost = round(xgb_pred, 2),
          bagged_tree = round(bag_pred, 2)
        )
      ))
      
    }, error = function(e){
      return(list(success = FALSE, error = paste("Prediction error:", e$message)))
    })
  }
}

# Define UI for application that draws a histogram
ui <- dashboardPage(
  
  dashboardHeader(title = "Listening Time Predictor"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Single Prediction", tabName = "single", icon = icon("microphone")),
      menuItem("Batch Prediction", tabName = "batch", icon = icon("table")),
      menuItem("Model Info", tabName = "model_info", icon = icon("info-circle"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
                        .content-wrapper, .right-side {
                          background-color: #f4f4f4;
                        }
                        .box {
                          border-radius: 8px;
                          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }"))
    ),
    
    tabItems(
      # Prediction Tab
      tabItem(tabName = "single",
              fluidRow(
                box(
                  title = "Input Parameters", status = "primary", solidHeader = TRUE, width = 6,
                  
                  # Inputs
                  selectInput("podcast_name", "Podcast:",
                              choices = c("Mystery Matters" = "Mystery Matters",
                                          "Joke Junction" = "Joke Junction",
                                          "Study Sessions" = "Study Sessions",
                                          "Digital Digest" = "Digital Digest",
                                          "Mind & Body" = "Mind & Body",
                                          "Fitness First" = "Fitness First",
                                          "Criminal Minds" = "Criminal Minds",
                                          "News Roundup" = "News Roundup",
                                          "Daily Digest" = "Daily Digest",
                                          "Music Matters" = "Music Matters"),
                              selected = "Mystery Matters"
                  ),
                  
                  numericInput("episode_title", "Episode:",
                              value = 50, min = 0, max = 1000),
                  
                  numericInput("episode_length_minutes", "Episode Length:",
                               value = 64.50, min = 0, max = 350.50),
                  
                  selectInput("genre", "Genre:",
                              choices = c(
                                "True Crime" = "True Crime",
                                "Comedy" = "Comedy",
                                "Education" = "Education",
                                "Technology" = "Technology",
                                "Health" = "Health",
                                "News" = "News",
                                "Music" = "Music",
                                "Sports" = "Sports",
                                "Business" = "Business",
                                "Lifestyle" = "Lifestyle"),
                              selected = "True Crime"
                  ),
                  
                  numericInput("host_popularity_percentage", "Host Popularity:",
                               value = 1.0, min = 60.5, max = 150.0),
                  
                  selectInput("publication_day", "Day:",
                              choices = c(
                                "Thursday" = "Thursday",
                                "Saturday" = "Saturday",
                                "Tuesday" = "Tuesday",
                                "Monday" = "Monday",
                                "Sunday" = "Sunday",
                                "Wednesday" = "Wednesday",
                                "Friday" = "Friday"),
                              selected = "Thursday"
                  ),
                  
                  selectInput("publication_time", "Time:",
                              choices = c(
                                "Night" = "Night",
                                "Afternoon" = "Afternoon",
                                "Evening" = "Evening",
                                "Morning" = "Morning"),
                              selected = "Night"
                  ),
                  
                  selectInput("episode_sentiment", "Sentiment:",
                              choices = c(
                                "Positive" = "Positive",
                                "Negative" = "Negative",
                                "Neutral" = "Neutral"),
                              selected = "Positive"
                  ),
                  
                  numericInput("number_of_ads", "No. of Ads:",
                               value = 0.0, min = 1.5, max = 110.0),
                  
                  numericInput("guest_popularity_percentage", "Guest Popularity:",
                               value = 0.0, min = 55.5, max = 120.0),
                  
                  br(),
                  actionButton("predict_single", "Predict Listening Time",
                               class = "btn-primary btn-lg", width = "100%")
                  
                ),
                
                # Ouput Box
                box(
                  title = "Prediction Results", status = "success", solidHeader = TRUE, width = 6,
                  
                  valueBoxOutput("prediction_box", width = 12),
                  
                  br(),
                  
                  h4("Model Confidence"),
                  verbatimTextOutput("model_confidence"),
                  
                  br(),
                  
                  # h4("Feature Importance"),
                  # plotlyOutput("feature_importance")
                  h4("Individual Model Predictions"),
                  verbatimTextOutput("individual_results"),
                  
                  br(),
                  
                  h4("Prediction Confidence"),
                  plotOutput("prediction_plot", height = "300px")
                )
              ),
              
              fluidRow(
                box(
                  title= "Input Summary", status = "info", solidHeader = TRUE,
                  width = 12,
                  DT::dataTableOutput("input_summary")
                )
              )
      ),
      
      # Prediction Tab
      tabItem(tabName = "batch",
              fluidRow(
                box(
                  title = "Batch Prediction", status = "primary", solidHeader = TRUE, width = 12,
                  
                  
                  h4("Upload CSV File"),
                  p("Upload a CSV file with the same structure as the training data"),
                  
                  fileInput("batch_file", "Choose CSV File", accept = c(".csv")),
                  checkboxInput("header", "Header", TRUE),
                  actionButton("predict_batch", "Predict Batch", class = "btn-primary"),
                  
                  br(),
                  downloadButton("download_predictions", "Download Predictions", class = "btn-success"),
                  
                  br(), br(),
                  
                  DT::dataTableOutput("batch_results")
                )
              )
      ),
      
      
      # Model Info Tab
      tabItem(tabName = "model_info",
              fluidRow(
                box(
                  title = "Model Architecture", status = "primary", solidHeader = TRUE, width = 6,
                  
                  h4("Ensemble Stacking Approach"),
                  p("This model uses an ensemble stacking approach with the following base models:"),
                  tags$ul(
                    tags$li("Random Forest (500 trees)"),
                    tags$li("XGBoost (500 trees, learning rate: 0.1)"),
                    tags$li("Bagged Decision Trees"),
                    tags$li("Tuned Decision Tree (meta-learner)")
                  ),
                  
                  h4("Data Preprocessing"),
                  tags$ul(
                    tags$li("Correlation filtering for numeric features"),
                    tags$li("Dummy encoding for categorical variables"),
                    tags$li("Normalization of numeric predictors"),
                    tags$li("Missing value imputation with mean")
                  )
                ),
                
                box(
                  title = "Model Performance", status = "primary", solidHeader = TRUE, width = 6,
                  
                  h4("Cross-Validation Results"),
                  verbatimTextOutput("model_metrics"),
                  
                  br(),
                  
                  h4("Model Comparison"),
                  DT::dataTableOutput("model_comparison")
                )
              )
      )
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  # load model once app launches
  ensemble_model <- reactive({
    tryCatch({
      load_ensemble_model()
    }, error = function(e){
      showNotification(paste("Error loading model:", e$message), type = "error")
      return(NULL)
    })
  })
  
  # Create predictor function
  predictor <- reactive({
    req(ensemble_model())
    create_shiny_predictor(ensemble_model())
  })
  
  # Store prediction results
  prediction_result <- reactiveVal(NULL) # Reactive Val. to store prediction
  
  # Create reactive input data frame
  current_input_df <- reactive({
    data.frame(
      id = round(runif(1, min = 0, max = 749999), 0),
      podcast_name = input$podcast_name,
      episode_title = paste0("Episode_", input$episode_title),
      episode_length_minutes = input$episode_length_minutes,
      genre = input$genre,
      host_popularity_percentage = input$host_popularity_percentage,
      publication_day = input$publication_day,
      publication_time = input$publication_time,
      guest_popularity_percentage = input$guest_popularity_percentage,
      number_of_ads = input$number_of_ads,
      episode_sentiment = input$episode_sentiment,
      stringsAsFactors = FALSE
    )
  })
  
  # Single prediction logic
  observeEvent(input$predict_single, {
    
    # Validate inputs first
    if(is.null(input$podcast_name) || input$podcast_name == "") {
      showNotification("Please select a podcast name", type = "warning")
      return()
    }
    
    # Get input dataframe
    input_df <- current_input_df()
    
    # Debug: Print input data structure
    cat("Input data structure:\n")
    str(input_df)
    
    # Make prediction
    tryCatch({
      result <- predictor()(input_df)
      
      if(result$success){
        prediction_result(result$ensemble_prediction) # update prediction store
        showNotification("Prediction completed successfully!", type = "message")
      } else {
        showNotification(paste("Prediction Failed:", result$error), type = "error")
        prediction_result(NULL)
      }
    }, error = function(e) {
      showNotification(paste("Error during prediction:", e$message), type = "error")
      prediction_result(NULL)
    })
  })
  # Error during prediction: 'arg' should be one of “default”, “message”, “warning”, “error”
  
  # prediction box output
  output$prediction_box <- renderValueBox({
    pred <- prediction_result()
    if (is.null(pred)) {
      valueBox(
        value = "Click Predict",
        subtitle = "Predicted Listening Time (minutes)",
        icon = icon("clock"),
        color = "light-blue"
      )
    } else {
      valueBox(
        value = paste(round(pred, 1), "min"),
        subtitle = "Predicted Listening Time",
        icon = icon("headphones"),
        color = "green"
      )
    }
  })
  
  # Get the full prediction result for other outputs
  full_prediction_result <- reactive({
    req(input$predict_single)  # Only run after prediction button is clicked
    
    input_df <- current_input_df()
    
    predictor()(input_df)
  })
  
  # Model confidence output
  output$model_confidence <- renderText({
    pred <- prediction_result()
    if (is.null(pred)) {
      "Make a prediction to see confidence metrics"
    } else {
      paste("Model RMSE: 13.3 minutes\n",
            "R-squared: 0.946\n",
            "Confidence Level: High\n",
            "Prediction Range: ±13.3 minutes")
    }
  })
  
  output$individual_results <- renderText({
    result <- full_prediction_result()
    if (!is.null(result) && result$success) {
      individual <- result$individual_predictions
      paste(
        "Random Forest:", individual$random_forest, "min\n",
        "XGBoost:", individual$xgboost, "min\n",
        "Bagged Tree:", individual$bagged_tree, "min"
      )
    } else {
      "No individual predictions available"
    }
  })
  
  # Prediction visualization
  output$prediction_plot <- renderPlot({
    result <- full_prediction_result()
    if (!is.null(result) && result$success) {
      individual <- result$individual_predictions
      df <- data.frame(
        Model = c("Random Forest", "XGBoost", "Bagged Tree", "Ensemble"),
        Prediction = c(individual$random_forest, individual$xgboost,
                       individual$bagged_tree, result$ensemble_prediction),
        Type = c(rep("Individual", 3), "Ensemble")
      )
      
      ggplot(df, aes(x = Model, y = Prediction, fill = Type)) +
        geom_col(alpha = 0.8) +
        scale_fill_manual(values = c("Individual" = "lightblue", "Ensemble" = "darkblue")) +
        theme_minimal() +
        labs(title = "Model Predictions Comparison",
             y = "Predicted Listening Time (minutes)",
             x = "Model") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        geom_text(aes(label = Prediction), vjust = -0.5, size = 3)
    } else {
      # Return empty plot if no prediction
      ggplot() +
        theme_void() +
        labs(title = "Make a prediction to see comparison")
    }
  })
  
  # Input summary Table
  output$input_summary <- DT::renderDataTable({
    # Always show current inputs, regardless of prediction status
    summary_data <- data.frame(
      Parameter = c(
        "Podcast Name", "Episode Title",
        "Episode Length (min)", "Genre",
        "Host Popularity (%)",
        "Publication Day", "Publication Time",
        "Guest Popularity (%)",
        "Number of Ads","Episode Sentiment"
      ),
      Value = c(
        input$podcast_name %||% "Not selected",
        input$episode_title %||% 0,
        input$episode_length_minutes %||% 0,
        input$genre %||% "Not selected",
        input$host_popularity_percentage %||% 0,
        input$publication_day %||% "Not selected",
        input$publication_time %||% "Not selected",
        input$guest_popularity_percentage %||% 0,
        input$number_of_ads %||% 0,
        input$episode_sentiment %||% "Not selected"
      )
    )
    
    DT::datatable(summary_data,
                  options = list(pageLength = 10, dom = 't', searching = FALSE),
                  rownames = FALSE)
  })
  
  # Model metrics output
  output$model_metrics <- renderText({
    "Ensemble Model Performance:\nTrained 0n a sample of 75,000 random records\n\nRMSE: 13.3 minutes\nR-squared: 0.946\nMAE: 10.9 minutes\n\nThis model explains 94.6% of the variance in listening time."
  })
  
  # Model comparison table
  output$model_comparison <- DT::renderDataTable({
    comparison_data <- data.frame(
      Model = c("Decision Tree", "Random Forest", "XGBoost", "Bagged Trees", "Ensemble"),
      RMSE = c(13.4, 13.9, 13.3, 13.7, 13.3),
      R_squared = c(0.757, 0.753, 0.760, 0.745, 0.946),
      Status = c("Base Model", "Base Model", "Base Model", "Base Model", "Final Model")
    )
    
    DT::datatable(comparison_data, options = list(pageLength = 10), rownames = FALSE)
  })
  
  # Batch prediction handling
  batch_data <- reactive({
    req(input$batch_file) # requires csv input
    
    tryCatch({
      read.csv(input$batch_file$datapath, header = input$header)
    }, error = function(e) {
      showNotification("Error reading file. Please check file format.", type = "error")
      return(NULL)
    })
  })
  
  batch_predictions <- eventReactive(input$predict_batch, {
    data <- batch_data()
    req(data)  # Ensure data is available
    
    withProgress(message = 'Making predictions...', value = 0, {
      predictions <- map_dfr(1:nrow(data), ~{
        incProgress(1/nrow(data), detail = paste("Row", .x))
        
        row_data <- data[.x, ]
        result <- predictor()(row_data)
        if(result$success){
          data.frame(
            row = .x,
            prediction = result$ensemble_prediction
          )
        } else {
          data.frame(  # FIXED: was "data.fram"
            row = .x,
            prediction = NA,
            error = result$error
          )
        }
      })
    })
    
    cbind(data, predicted_listening_time = predictions$prediction)
  })
  
  output$batch_results <- DT::renderDataTable({
    batch_predictions()
  }, options = list(scrollX = TRUE))
  
  # Download handler for batch predictions
  output$download_predictions <- downloadHandler(
    filename = function() {
      paste("listening_time_predictions_", Sys.chmod(), ".csv", sep = "")
    },
    content = function(file){
      data <- batch_predictions()
      if (!is.null(data)){
        write.csv(data, file, row.names = FALSE)
      }
    }
  )
}

# Run the application
shinyApp(ui = ui, server = server)

