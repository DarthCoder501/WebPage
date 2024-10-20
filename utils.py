# Import necessary libraries
import plotly.graph_objects as go  # For creating interactive visualizations

def create_gauge_chart(probability):
    """
    Create a gauge chart to visualize the churn probability.

    Args:
        probability (float): The churn probability value between 0 and 1.

    Returns:
        go.Figure: A Plotly Figure object representing the gauge chart.
    """
    # Determine the color of the gauge based on the probability value
    if probability < 0.3: 
        color = "green"  # Low probability of churn
    elif probability < 0.6:
        color = "yellow"  # Medium probability of churn
    else: 
        color = "red"  # High probability of churn

    # Create the gauge chart figure
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",  # Mode to display gauge and numeric value
            value=probability * 100,  # Convert probability to percentage for display
            domain={"x": [0, 1], "y": [0, 1]},  # Domain of the gauge within the figure
            title={"text": "Churn Probability", "font": {"size": 24, "color": "white"}},  # Title configuration
            number={"font": {"size": 40, "color": "white"}},  # Number font configuration
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},  # Axis range and tick color
                "bar": {"color": color},  # Sets the bar color based on the probability
                "bgcolor": "rgba(0, 0, 0, 0)",  # Transparent background for the gauge
                "borderwidth": 2,
                "bordercolor": "white",  # Border color of the gauge
                "steps": [
                    {"range": [0, 30], "color": "rgba(0, 255, 0, 0.3)"},  # Low probability step
                    {"range": [30, 60], "color": "rgba(255, 255, 0, 0.3)"},  # Medium probability step
                    {"range": [60, 100], "color": "rgba(255, 0, 0, 0.3)"}  # High probability step
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 100}  # Threshold line configuration
            }
        )
    )

    # Update the layout of the gauge chart
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Set background color to transparent
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Set plot area background to transparent
        font={"color": "white"},  # Font color for text in the chart
        width=400,  # Width of the figure
        height=300,  # Height of the figure
        margin=dict(l=20, r=20, t=50, b=50)  # Set margins for the figure
    )

    return fig  # Return the created figure


def create_model_probability_chart(probabilities):
    """
    Create a horizontal bar chart to visualize churn probabilities for different models.

    Args:
        probabilities (dict): A dictionary with model names as keys and their respective probabilities as values.

    Returns:
        go.Figure: A Plotly Figure object representing the bar chart.
    """
    # Extract model names and their probabilities
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    # Create a horizontal bar chart figure
    fig = go.Figure(data=[
        go.Bar(
            x=probs,  # Set the x-axis to probabilities
            y=models,  # Set the y-axis to model names
            orientation="h",  # Horizontal bar chart
            text=[f"{p:.2f}%" for p in probs],  # Display probabilities as text on bars
            textposition="auto",  # Position text automatically
        )
    ])

    # Update the layout of the bar chart
    fig.update_layout(
        title="Churn Probability by Model",  # Chart title
        yaxis_title="Models",  # Y-axis title
        xaxis_title="Probability",  # X-axis title
        xaxis=dict(tickformat=".0%", range=[0, 1]),  # Format x-axis ticks as percentages and set range
        height=400,  # Height of the figure
        margin=dict(l=20, r=20, t=40, b=20)  # Set margins for the figure
    )

    return fig  # Return the created figure
