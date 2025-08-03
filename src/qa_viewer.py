#!/usr/bin/env python3

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import pandas as pd
import glob
import os
import argparse
from datetime import datetime

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "QA Pairs Viewer"

# Global variable to store the specified CSV file
SPECIFIED_CSV = None

def get_csv_file():
    """Return the specified CSV file"""
    if SPECIFIED_CSV and os.path.exists(SPECIFIED_CSV):
        return SPECIFIED_CSV
    else:
        print(f"Error: Specified CSV file '{SPECIFIED_CSV}' not found.")
        return None

def load_csv_data(filename):
    """Load and return CSV data"""
    if not filename or not os.path.exists(filename):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filename)
        # Add index for selection
        df['id'] = range(len(df))
        return df
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()

# Initialize layout
def create_layout():
    return html.Div([
        html.H1("QA Pairs Data Viewer", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        
        # Search section
        html.Div([
            html.Div([
                html.Label("Search:", style={'fontWeight': 'bold', 'marginRight': 10}),
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder='Search topics or questions...',
                    style={'width': '300px', 'padding': '5px'}
                )
            ], style={'display': 'inline-block'})
        ], style={'marginBottom': 20, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
        
        # Data table
        html.Div([
            html.H3("QA Pairs Overview", style={'color': '#2c3e50'}),
            dash_table.DataTable(
                id='qa-table',
                columns=[
                    {'name': 'ID', 'id': 'id', 'type': 'numeric'},
                    {'name': 'Topic', 'id': 'topic', 'type': 'text'},
                    {'name': 'Question', 'id': 'question', 'type': 'text'},
                ],
                data=[],
                row_selectable='single',
                selected_rows=[],
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'id'}, 'width': '50px'},
                    {'if': {'column_id': 'topic'}, 'width': '200px'},
                    {'if': {'column_id': 'question'}, 'width': '400px'},
                ],
                style_data={
                    'backgroundColor': '#f8f9fa',
                    'border': '1px solid #dee2e6'
                },
                style_header={
                    'backgroundColor': '#343a40',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        ], style={'marginBottom': 30}),
        
        # Detailed view
        html.Div(id='detailed-view', style={'marginTop': 20})
    ])

# Layout will be set after CSV is specified

@app.callback(
    [Output('qa-table', 'data'),
     Output('qa-table', 'selected_rows')],
    [Input('search-input', 'value')]
)
def update_table(search_value):
    csv_file = get_csv_file()
    if not csv_file:
        return [], []
    
    df = load_csv_data(csv_file)
    if df.empty:
        return [], []
    
    # Apply search filter
    if search_value:
        mask = (
            df['topic'].str.contains(search_value, case=False, na=False) |
            df['question'].str.contains(search_value, case=False, na=False)
        )
        df = df[mask]
    
    # Return only the columns needed for the table
    table_data = df[['id', 'topic', 'question']].to_dict('records')
    return table_data, []

@app.callback(
    Output('detailed-view', 'children'),
    [Input('qa-table', 'selected_rows')]
)
def update_detailed_view(selected_rows):
    if not selected_rows:
        return html.Div([
            html.H3("Select a QA pair to view details", 
                   style={'textAlign': 'center', 'color': '#6c757d', 'marginTop': 50})
        ])
    
    csv_file = get_csv_file()
    if not csv_file:
        return html.Div("No data available")
    
    df = load_csv_data(csv_file)
    if df.empty:
        return html.Div("No data available")
    
    row_idx = selected_rows[0]
    # Get the actual row by matching the id
    table_data = df.to_dict('records')
    if row_idx >= len(table_data):
        return html.Div("Invalid selection")
    
    row = table_data[row_idx]
    
    return html.Div([
        html.H3("Detailed View", style={'color': '#2c3e50', 'borderBottom': '2px solid #2c3e50', 'paddingBottom': 10}),
        
        # Topic and Question
        html.Div([
            html.H4("Topic:", style={'color': '#495057', 'marginBottom': 5}),
            html.P(row['topic'], style={'fontSize': 16, 'marginBottom': 15, 'padding': '10px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'}),
            
            html.H4("Question:", style={'color': '#495057', 'marginBottom': 5}),
            html.P(row['question'], style={'fontSize': 16, 'marginBottom': 15, 'padding': '10px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'}),
            
            html.H4("Grading Notes:", style={'color': '#495057', 'marginBottom': 5}),
            html.Pre(row['grading_notes'], style={'fontSize': 14, 'marginBottom': 20, 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #dee2e6'})
        ]),
        
        # Answers
        html.H4("Generated Answers:", style={'color': '#2c3e50', 'marginTop': 20, 'marginBottom': 15}),
        
        html.Div([
            # Complete Answer
            html.Div([
                html.H5("✅ Complete Answer (All Grading Criteria)", style={'color': '#28a745', 'marginBottom': 10}),
                html.P(row['complete_answer'], style={'padding': '15px', 'backgroundColor': '#d4edda', 'borderRadius': '5px', 'border': '1px solid #c3e6cb'})
            ], style={'marginBottom': 20}),
            
            # Modified Answer
            html.Div([
                html.H5("⚠️ Modified Answer (With Strategic Omissions)", style={'color': '#ffc107', 'marginBottom': 10}),
                html.P(row['modified_answer'], style={'padding': '15px', 'backgroundColor': '#fff3cd', 'borderRadius': '5px', 'border': '1px solid #ffeaa7'})
            ], style={'marginBottom': 20}),
            
            # Changes Made
            html.Div([
                html.H5("📝 Changes Made:", style={'color': '#6c757d', 'marginBottom': 10}),
                html.P(row['changes_made'], style={'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px', 'border': '1px solid #dee2e6', 'fontStyle': 'italic'})
            ], style={'marginBottom': 20})
        ])
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QA Pairs Data Viewer")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file to view")
    args = parser.parse_args()
    
    # Set the global CSV file
    SPECIFIED_CSV = args.csv
    
    # Set the layout after CSV is specified
    app.layout = create_layout()
    
    print("Starting QA Pairs Viewer...")
    print(f"Loading CSV file: {SPECIFIED_CSV}")
    print("Open http://localhost:8060 in your browser")
    app.run(debug=True, host='localhost', port=8060)