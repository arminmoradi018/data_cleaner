from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def server(input, output, session):
    
    raw_data = reactive.value(pd.DataFrame())
    cleaned_data = reactive.value(pd.DataFrame())
    analysis_data = reactive.value(pd.DataFrame())
    desc_data = reactive.value(pd.DataFrame())

    
    # Sidebar
    @reactive.Effect
    def update_selectors():
        df = raw_data()
        if df.empty:
            ui.update_selectize("cols_to_drop", choices=[])
            ui.update_selectize("cols_to_transform", choices=[])
            return
        
        ui.update_selectize("cols_to_drop", choices=list(df.columns))
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        ui.update_selectize("cols_to_transform", choices=numeric_cols)
    
    
    @reactive.Effect
    @reactive.event(input.file)
    def handle_upload():
        try:
            file = input.file()[0]
            if not file["name"].endswith(".csv"):
                raise ValueError("Only CSV files are supported")
            
            df = pd.read_csv(file["datapath"])
            raw_data.set(df)
            cleaned_data.set(df.copy())
            analysis_data.set(pd.DataFrame())
            
            ui.notification_show("File uploaded successfully!", duration=3, type="message")
            
        except Exception as e:
            ui.notification_show(f"Error loading file: {str(e)}", duration=5, type="error")
            raw_data.set(pd.DataFrame())
            cleaned_data.set(pd.DataFrame())
            analysis_data.set(pd.DataFrame())
    
    
    @reactive.Effect
    @reactive.event(input.analyze)
    def analyze_data():
        if raw_data().empty:
            ui.notification_show("Please upload a file first!", duration=3, type="warning")
            return
        
        df = raw_data()
        analysis = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isna().sum(),
            "Unique Values": df.nunique()
        }).sort_values("Missing Values", ascending=False)

        desc = (
            df.describe().T.reset_index()
            .rename(columns={"index": "Column"})
        )
        
        analysis_data.set(analysis)
        desc_data.set(desc)
        ui.notification_show("Analysis completed!", duration=3, type="message")
    
    
    @reactive.Effect
    @reactive.event(input.clean)
    def clean_data():
        if raw_data().empty:
            ui.notification_show("No data to clean!", duration=3, type="warning")
            return
        
        df = raw_data().copy()
        
        
        cols_to_drop = input.cols_to_drop()
        if cols_to_drop:
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        
        na_action = input.na_action()
        if na_action != "no_change":
            numeric_cols = df.select_dtypes(include=np.number).columns
            
            if na_action == "zero":
                df = df.fillna(0)
            elif na_action == "mean" and not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif na_action == "median" and not numeric_cols.empty:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif na_action == "drop":
                df = df.dropna()
        
        
        transform_method = input.transform_method()
        cols_to_transform = input.cols_to_transform()
        
        if transform_method != "none" and cols_to_transform:
            numeric_cols = df.select_dtypes(include=np.number).columns
            cols_to_transform = [col for col in cols_to_transform if col in numeric_cols]

            
            if cols_to_transform:
                if transform_method == "normalize":
                    scaler = MinMaxScaler()
                    df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
                elif transform_method == "standardize":
                    scaler = StandardScaler()
                    df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
        cleaned_data.set(df)
        ui.notification_show("Data cleaning applied!", duration=3, type="message")
    
    
    @reactive.Effect
    @reactive.event(input.reset)
    def reset_data():
        if raw_data().empty:
            ui.notification_show("No data to reset!", duration=3, type="warning")
            return
        
        cleaned_data.set(raw_data().copy())
        analysis_data.set(pd.DataFrame())
        desc_data.set(pd.DataFrame())
        ui.update_selectize("cols_to_drop", selected=[])
        ui.update_select("na_action", selected="no_change")
        ui.update_selectize("cols_to_transform", selected=[])
        ui.update_select("transform_method", selected="none")
        ui.notification_show("Data reset to original state", duration=3, type="message")
    
    
    @render.download(filename="cleaned_data.csv")
    def download():
        df = cleaned_data() if not cleaned_data().empty else pd.DataFrame()
        with BytesIO() as buf:
            df.to_csv(buf, index=False)
            yield buf.getvalue()
    
    # Analysis tables
    @render.ui
    def data_table_container():
        df = cleaned_data()
        if df.empty:
            return ui.div(
                ui.h4("Data Preview"),
                ui.markdown("Please upload a CSV file to begin."),
                style="min-height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center;"
            )
        return ui.output_data_frame("data_table")
    
    @render.data_frame
    def data_table():
        return cleaned_data()
    
    @render.ui
    def desc_table_container():
        df = desc_data()
        return ui.output_data_frame("desc_table")

    @render.data_frame
    def desc_table():
        return desc_data()
    @render.ui
    def columns_info_container():
        df = analysis_data()
        if df.empty:
            return ui.div(
                ui.h4("Data Analysis"),
                ui.markdown("Click 'Analyze' to see data statistics."),
                style="min-height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center;"
            )
        return ui.output_data_frame("analysis_table")
    
    @render.data_frame
    def analysis_table():
        return analysis_data()



    # Visualization
    @reactive.Effect
    def update_feature_and_target_choices():
        """Update feature & target select choices from raw_data (not cleaned)."""
        df = raw_data()
        if df is None or df.empty:
            ui.update_selectize("features", choices=[])
            ui.update_select("target_col", choices=[])
        else:
            cols = list(df.columns)
            ui.update_selectize("features", choices=cols)
            ui.update_select("target_col", choices=cols)
    @render.plot
    @reactive.event(input.generate_plot , input.use_cleaned)
    def viz_plot():
        """Generate visualization using selected features. Uses raw_data or cleaned_data based on toggle."""
        df_src = cleaned_data() if input.use_cleaned() else raw_data()
        if df_src is None or df_src.empty:
            # no data -> show friendly message plot
            plt.figure()
            plt.text(0.5, 0.5, "Please upload a CSV file first.", ha="center", va="center", fontsize=12)
            plt.axis("off")
            return plt.gcf()

        # Get inputs (safe)
        features = list(input.features() or [])
        plot_type = input.plot_type() or "scatter"
        target = input.target_col() or None

        # Validate features existence in dataframe (may have been removed)
        features = [f for f in features if f in df_src.columns]
        if not features:
            plt.figure()
            plt.text(0.5, 0.5, "No valid features selected (they may have been removed).", ha="center", va="center", fontsize=12)
            plt.axis("off")
            return plt.gcf()

        # Palette (fixed)
        palette = ["#1abc9c", "#3498db", "#e67e22"]
        sns.set_palette(palette)

        try:
            n = len(features)
            # For pairplot we want larger square; for row plots we scale by n
            if plot_type == "pairwise":
                fig_size = max(6, 2.5 * n) 
                sns.set_theme(style="white", font_scale=0.8)
                cols_to_plot = features.copy()
                if target and target in df_src.columns and target not in cols_to_plot:
                    cols_to_plot.append(target)

                # create pairplot (hue only if target specified)
                pairplot = sns.pairplot(
                    data=df_src[cols_to_plot],
                    hue=target if (target and target in df_src.columns) else None,
                    palette=palette if (target and target in df_src.columns) else None,
                    diag_kind="kde",
                    corner=False,
                    plot_kws={"s": 45, "alpha": 0.85, "edgecolor": "w", "linewidth": 0.4}
                )
                # adjust size & layout
                pairplot.figure.set_size_inches(fig_size, fig_size)
                pairplot.figure.subplots_adjust(top=0.94, bottom= 0.08, left= 0.085, right=0.92, wspace=0.12, hspace=0.12)

                # adjust legend
                if getattr(pairplot, "_legend", None) is not None:
                    pairplot._legend.set_bbox_to_anchor((1.007, 0.9))
                    pairplot._legend.set_frame_on(True)
                    pairplot._legend.set_title("Target") 
                    for text in pairplot._legend.get_texts():
                        text.set_fontsize(7)

                return pairplot.figure

            elif plot_type == "hist":
                fig_w = max(6, 3.0 * n)
                fig_h = 4
                fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
                axes = axes[0]
                sns.set_theme(style="whitegrid", font_scale=1.0)
                for ax, col in zip(axes, features):
                    sns.histplot(df_src[col].dropna(), kde=True, ax=ax, color=palette[1], edgecolor="k", linewidth=0.3)
                plt.tight_layout()
                return fig

            elif plot_type == "box":
                # side-by-side boxplots
                fig_w = max(6, 3.0 * n)
                fig_h = 4.5
                fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
                axes = axes[0]
                sns.set_theme(style="whitegrid", font_scale=1.0)
                for ax, col in zip(axes, features):
                    if target and target in df_src.columns:
                        sns.boxplot(x=target, y=col, data=df_src, ax=ax, palette=palette)
                    else:
                        sns.boxplot(y=df_src[col].dropna(), ax=ax, color=palette[2])
                         
                
                plt.tight_layout()
                return fig

            else:
                # default: scatter plot using first two features (if available)
                if n < 2:
                    plt.figure()
                    plt.text(0.5, 0.5, "Select at least two features for scatter.", ha="center", va="center")
                    plt.axis("off")
                    return plt.gcf()

                fig_w = max(6, 5)
                fig_h = 4.5
                plt.figure(figsize=(fig_w, fig_h))
                sns.scatterplot(
                    data=df_src,
                    x=features[0],
                    y=features[1],
                    hue=target if (target and target in df_src.columns) else None,
                    palette=palette if (target and target in df_src.columns) else None,
                    s=60,
                    alpha=0.85,
                    edgecolor="w",
                    linewidth=0.4
                )
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.tight_layout()
                return plt.gcf()

        except Exception as exc:
            # any plotting error -> show message on canvas
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, f"Error drawing plot:\n{str(exc)}", ha="center", va="center", wrap=True)
            plt.axis("off")
            return plt.gcf()
        

        
    
    # Dimensionality Reduction
    @reactive.Effect
    def update_color_choices():
        """Update target column options."""
        df = raw_data()
        if df is None or df.empty:
            ui.update_select("color_col", choices=[])
        else:
            ui.update_select("color_col", choices=list(df.columns))

    @render.ui
    def perplexity_ui():
        method = input.reduction_method()
        if method == "tsne":
            return ui.input_slider(
                "perplexity",
                "Perplexity",
                min=10,
                max=50,
                value=30,
                width="100%"
            )
        else:
            return None        
    @render.plot
    @reactive.event(input.run_reduction , input.use_cleaned_reduction)
    def reduction_plot():
        """Apply PCA or t-SNE and visualize components."""
        df_src = cleaned_data() if input.use_cleaned_reduction() else raw_data()

        if df_src is None or df_src.empty:
            plt.figure()
            plt.text(0.5, 0.5, "Please upload a dataset first.", ha="center", va="center")
            plt.axis("off")
            return plt.gcf()

        num_cols = df_src.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) < 2:
            plt.figure()
            plt.text(0.5, 0.5, "Not enough numeric features for reduction.", ha="center", va="center")
            plt.axis("off")
            return plt.gcf()

        X = df_src[num_cols].dropna()
        if X.empty:
            plt.figure()
            plt.text(0.5, 0.5, "No valid numeric data for reduction.", ha="center", va="center")
            plt.axis("off")
            return plt.gcf()
        X_scaled = StandardScaler().fit_transform(X)
        method = input.reduction_method()
        n_components = input.n_components()
        target_col = input.color_col()

        try:
            if method == "pca":
                model = PCA(n_components=n_components, random_state=42)
                components = model.fit_transform(X_scaled)
                var_ratio = model.explained_variance_ratio_
                var_text = f"Explained Variance: {np.sum(var_ratio):.2f}"
            else:
                perplexity = input.perplexity() if "perplexity" in input else 30

                model = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=perplexity,
                    learning_rate="auto",
                    init="pca"
                )
                components = model.fit_transform(X_scaled)
                var_text = f"(Perplexity = {perplexity})"

            comp_cols = [f"Component {i+1}" for i in range(n_components)]
            df_red = pd.DataFrame(components, columns=comp_cols, index=X.index)

            if target_col and target_col in df_src.columns:
                df_red[target_col] = df_src.loc[X.index, target_col].values

            sns.set_theme(style="whitegrid", font_scale=1.0)
            palette = sns.color_palette("viridis")

            if n_components == 2:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.scatterplot(
                    data=df_red,
                    x="Component 1", y="Component 2",
                    hue=target_col if target_col else None,
                    palette=palette if target_col else None,
                    s=60, alpha=0.85, edgecolor="w", linewidth=0.4
                )
                ax.set_title(f"{method.upper()} - 2D Projection {var_text}", fontsize=12, fontweight="bold")
                plt.tight_layout()
                return fig

            elif n_components == 3:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                sc = ax.scatter(
                    df_red["Component 1"], df_red["Component 2"], df_red["Component 3"],
                    c=None if not target_col else pd.factorize(df_red[target_col])[0],
                    cmap="viridis", s=50, alpha=0.85, edgecolors="w", linewidths=0.3
                )
                ax.set_title(f"{method.upper()} - 3D Projection {var_text}", fontsize=12, fontweight="bold")
                plt.tight_layout()
                return fig

        except Exception as exc:
            plt.figure()
            plt.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center", wrap=True)
            plt.axis("off")
            return plt.gcf()