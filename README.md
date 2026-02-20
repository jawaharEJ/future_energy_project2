# Future Energy Project

Minimal notes to deploy this Flask app to Render.

Deployment steps (Render Web Service):

1. Connect your GitHub repo to Render and create a new **Web Service**.
2. Set the **Build Command** (optional):

```bash
pip install -r requirements.txt
```

3. Set the **Start Command**:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

4. Add environment variables in Render dashboard:
   - `SECRET_KEY` — set to a secure random string.

5. Ensure `runtime.txt` matches the Python version you want.

Notes:
- Gunicorn is included in `requirements.txt` for Render. Locally on Windows, run `python app.py`.
- I added minimal templates to avoid TemplateNotFound errors; expand them as needed.

If you want, I can create a small `.env.example` and a Git commit with these changes.
# Future Energy Consumption Predictor

A Flask-based web application that predicts future energy consumption using machine learning. Features user authentication, admin dashboard, chatbot, and data visualization.

## Features

- **User Registration & Login**: Secure user accounts with session management.
- **Energy Prediction**: Uses Linear Regression to predict consumption based on year, month, population, and industrial growth.
- **Data Visualization**: Generates charts showing historical data and predictions using Matplotlib.
- **Admin Dashboard**: View user statistics and manage data.
- **Chatbot**: Interactive bot answering questions about the app.
- **Responsive UI**: Professional interface built with Bootstrap.

## Technologies Used

- **Backend**: Flask (Python)
- **ML**: scikit-learn (Linear Regression)
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Data**: Pandas, Matplotlib
- **Storage**: JSON for users (can be upgraded to database)

## Installation

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `python app.py`
6. Open http://127.0.0.1:5000/register to register, then login.

## Usage

- Register a new account or login.
- Enter prediction parameters and click Predict.
- View the chart and result.
- Access chatbot for help.
- Admins can view dashboard at /admin.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: CSS and generated images
- `energy_data.csv`: Sample data
- `users.json`: User data storage
- `requirements.txt`: Dependencies

## Future Enhancements

- Database integration
- API endpoints
- Real-time features
- Advanced ML models
- Deployment

## License

This project is for educational purposes.