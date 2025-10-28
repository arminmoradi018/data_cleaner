from shiny import ui
app_ui = ui.page_fluid(
    ui.panel_title("Data Cleaner", window_title="Data Cleaner"),
    ui.page_sidebar(
        ui.sidebar(     
                ui.div(
                    ui.input_file(
                    id="file",
                    label="Upload CSV File",
                    accept=".csv",
                    multiple=False,
                    width="100%"
                    ),
                    ui.input_action_button(
                    id="analyze",
                    label="Analyze",
                    class_="btn-outline-dark",
                    width="100%",
                    style="margin-top: 0.5rem;"
                    )
                ),       
                        
                ui.hr(),
                ui.markdown("**Remove Columns**"),
                ui.input_selectize(
                    id="cols_to_drop",
                    label=None,
                    choices=[],
                    multiple=True,
                    width="100%"
                ),
                                
                ui.hr(),
                ui.markdown("**With NaNs:**"),
                ui.input_select(
                    id="na_action",
                    label=None,
                    choices={
                        "no_change": "No change",
                        "zero": "Replace with 0",
                        "mean": "Replace with column mean",
                        "median": "Replace with column median",
                        "drop": "Drop rows with missing values"
                    },
                    selected="no_change",
                    width="100%"
                ),
                                
                ui.hr(),
                ui.markdown("**Columns to transform**"),
                ui.input_selectize(
                    id="cols_to_transform",
                    label=None,
                    choices=[],
                    multiple=True,
                    width="100%"
                ),

                ui.hr(),
                ui.markdown("**Transform Strategy**"),
                ui.input_select(
                    id="transform_method",
                    label=None,
                    choices={
                        "none": "No transformation",
                        "normalize": "Normalization (0-1)",
                        "standardize": "Standardization (0 mean, 1 variance)"
                    },
                    selected="none",
                    width="100%"
                ),

                ui.hr(),
                ui.markdown("**Actions**"),
                ui.div(
                            
                    ui.input_action_button(
                        id="clean",
                        label="Clean",
                        class_="btn-outline-dark",
                        width="100%"
                    ),
                    ui.download_button(
                        id="download",
                        label="Download Cleaned Data",
                        class_="btn-outline-dark",
                        width="100%"
                    ),
                    ui.input_action_button(
                        id="reset",
                        label="Reset",
                        class_="btn-outline-dark",
                        width="100%"
                    ),
                            
                    style="gap: 15px; display: flex; flex-direction: column;"
                ),
                        
                ui.hr(),
                ui.input_dark_mode(id="dark_mode", mode="light"),
                        
                width=300,
                style="height:100%;"
        ),
    

        ui.navset_bar(
            ui.nav_panel(
                "Data",
                ui.div(
                    ui.output_ui("data_table_container")
                )
            ),
            ui.nav_panel(
                "Analysis",
                ui.div(
                    ui.output_ui("desc_table_container"),
                    ui.br(),
                    ui.output_ui("columns_info_container"),
                    style="gap: 20px; display: flex; flex-direction: column;"
                )
            ),
            ui.nav_panel(
                "Visualization",
                ui.div(
                    # container with two columns using flexbox (left: controls, right: plot)
                    ui.div(
                        # left column: controls (fixed width)
                        ui.card(
                            ui.h4("Visualization Controls"),
                            ui.input_selectize(
                                "features",
                                "Select Features",
                                choices=[],
                                multiple=True,
                                width="100%"
                            ),
                            ui.input_select(
                                "target_col",
                                "Color by (Target)",
                                choices=[],
                                width="100%"
                            ),
                            ui.input_select(
                                "plot_type",
                                "Plot Type",
                                choices={
                                    "scatter": "Scatter Plot",
                                    "pairwise": "Pairwise Relationships",
                                    "hist": "Histogram",
                                    "box": "Box Plot"
                                },
                                selected="scatter",
                                width="100%"
                            ),
                            ui.input_action_button(
                                "generate_plot",
                                "Generate Visualization",
                                class_="btn-outline-dark",
                                width="100%"
                            ),
                            ui.hr(),
                            ui.input_checkbox("use_cleaned", "Use Cleaned Data", value=False),
                            style="width:250px; min-width:250px; padding:5px; box-sizing:border-box;"
                        ),

                        # right column: plot output (flexible)
                        ui.div(
                            ui.card(
                                ui.output_plot("viz_plot", height= "530Px"),
                                style=" height:100%;"
                            ),
                            style="flex:1; box-sizing:border-box;"
                        ),

                        style="display:flex; gap:16px; align-items:flex-start; width:100%;"
                    ),
                    style="width:100%;"
                )
            ),
            ui.nav_panel(
                "Dimensionality Reduction",
                ui.div(
                    ui.div(
                        # Left controls
                        ui.card(
                            ui.h4("Reduction Controls"),
                            ui.input_select(
                                "reduction_method",
                                "Select Method",
                                choices={"pca": "PCA", "tsne": "t-SNE"},
                                selected="pca",
                                width="100%"
                            ),
                            ui.output_ui("perplexity_ui"),
                            ui.input_numeric(
                                "n_components",
                                "Number of Components",
                                value=2,
                                min=2,
                                max=3,
                                width="100%"
                            ),
                            ui.input_select(
                                "color_col",
                                "Color by (Target)",
                                choices=[],
                                width="100%"
                            ),
                            ui.input_checkbox("use_cleaned_reduction", "Use Cleaned Data", value=False),
                            ui.input_action_button(
                                "run_reduction",
                                "Run Reduction",
                                class_="btn-outline-dark",
                                width="100%"
                            ),
                            style="width:250px; min-width:250px; padding:5px; box-sizing:border-box;"
                        ),

                        # Right plot
                        ui.div(
                            ui.card(
                                ui.output_plot("reduction_plot", height="530px"),
                                style="height:100%;"
                            ),
                            style="flex:1; box-sizing:border-box;"
                        ),

                        style="display:flex; gap:16px; align-items:flex-start; width:100%;"
                    ),
                    style="width:100%;"
                )
            ),

            title=""         
        ),
        
    ),
    ui.tags.style("""
        /* ==============================
        Dark mode fix for buttons & tables
        ============================== */

        [data-bs-theme="dark"] .btn-outline-dark {
            color: #e0e0e0 !important;
            border-color: #d0d0d0 !important;
            background-color: transparent !important;
        }
        
        [data-bs-theme="dark"] .shiny-output-output table thead th,
        [data-bs-theme="dark"] .shiny-html-output table thead th {
            
            background-color: #151313 !important;
            color: #eaeaea !important;                
            
        }
        """
    )    
)