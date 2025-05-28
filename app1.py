from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

ERP_API_BASE_URL = "https://your-erp-api.com/v1"
ERP_API_KEY = "your-api-key"


class ERPChatbot:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        
        # Collect all patterns from intents and subintents + FAQ questions
        import json
        all_text = []
        
        # Load knowledge graph and collect patterns
        with open('knowledge_graph.json') as f:
            self.knowledge_graph = json.load(f)
            for intent in self.knowledge_graph['intents']:
                all_text.extend(intent['patterns'])
                if 'subintents' in intent:
                    for sub in intent['subintents']:
                        all_text.extend(sub['patterns'])
        
        # Load FAQs and add questions to patterns
        self.faqs = pd.read_csv('erp_faqs.csv')
        all_text.extend(self.faqs['question'].tolist())
        
        # Fit vectorizer once with all text
        self.vectorizer.fit(all_text)
        
        # Store vectorized patterns for each intent
        for intent in self.knowledge_graph['intents']:
            intent['patterns_embeddings'] = self.vectorizer.transform(intent['patterns'])
            if 'subintents' in intent:
                for sub in intent['subintents']:
                    sub['patterns_embeddings'] = self.vectorizer.transform(sub['patterns'])
        
        # Initialize database and API session
        self.init_db()
        
        self.erp_session = requests.Session()
        self.erp_session.headers.update({
            'Authorization': f'Bearer {ERP_API_KEY}',
            'Content-Type': 'application/json'
        })



    def init_db(self):
        conn = sqlite3.connect('erp_chatbot.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (
                         id
                         INTEGER
                         PRIMARY
                         KEY
                         AUTOINCREMENT,
                         user_id
                         TEXT,
                         message
                         TEXT,
                         response
                         TEXT,
                         intent
                         TEXT,
                         subintent
                         TEXT,
                         sentiment
                         TEXT,
                         confidence
                         REAL,
                         timestamp
                         DATETIME
                     )''')
        c.execute('''CREATE TABLE IF NOT EXISTS api_calls
                     (
                         id
                         INTEGER
                         PRIMARY
                         KEY
                         AUTOINCREMENT,
                         conversation_id
                         INTEGER,
                         endpoint
                         TEXT,
                         parameters
                         TEXT,
                         response_code
                         INTEGER,
                         timestamp
                         DATETIME
                     )''')
        conn.commit()
        conn.close()

    def analyze_sentiment(self, text):
        lower = text.lower()
        if any(word in lower for word in ['good', 'great', 'excellent', 'happy']):
            return {'label': 'POSITIVE', 'score': 0.9}
        elif any(word in lower for word in ['bad', 'terrible', 'sad', 'angry']):
            return {'label': 'NEGATIVE', 'score': 0.9}
        return {'label': 'NEUTRAL', 'score': 0.6}

    def classify_intent(self, text):
        text_embedding = self.vectorizer.transform([text.lower()])
        best_intent = None
        highest_sim = 0
        for intent in self.knowledge_graph['intents']:
            similarities = cosine_similarity(text_embedding, intent['patterns_embeddings'])
            max_sim = np.max(similarities)
            if max_sim > highest_sim:
                highest_sim = max_sim
                best_intent = intent

        subintent = None
        highest_sub_sim = 0
        if best_intent and 'subintents' in best_intent:
            for sub in best_intent['subintents']:
                similarities = cosine_similarity(text_embedding, sub['patterns_embeddings'])
                sub_sim = np.max(similarities)
                if sub_sim > highest_sub_sim:
                    highest_sub_sim = sub_sim
                    subintent = sub['subintent']

        return {
            'intent': best_intent['intent'] if best_intent else None,
            'subintent': subintent,
            'confidence': float(max(highest_sim, highest_sub_sim))
        }

    def call_erp_api(self, endpoint, params=None, method='GET'):
        try:
            if method == 'POST':
                response = self.erp_session.post(f"{ERP_API_BASE_URL}/{endpoint}", json=params or {})
            else:
                response = self.erp_session.get(f"{ERP_API_BASE_URL}/{endpoint}", params=params or {})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def generate_response(self, intent_data, user_message):
        intent = intent_data['intent']
        subintent = intent_data['subintent']
        matched_intent = next((i for i in self.knowledge_graph['intents'] if i['intent'] == intent), None)
        if not matched_intent:
            return self.get_fallback_response(user_message)

        if intent == "organizational_structure":
            return self.handle_organizational_structure(subintent)
        if intent == "expenses":
            return self.handle_expense_queries(subintent)
        if intent == "travel_management":
            return self.handle_travel_queries(subintent)
        if intent == "payroll_queries":
            return self.handle_payroll_queries(subintent)
        if intent == "attendance":
            return self.handle_attendance_queries(subintent)
        if intent == "auto_invoice":
            return self.handle_auto_invoice_queries(subintent)
        if intent == "leave_management":
            return self.handle_leave_management(subintent)

        if subintent and 'subintents' in matched_intent:
            matched_subintent = next((s for s in matched_intent['subintents'] if s['subintent'] == subintent), None)
            if matched_subintent:
                return matched_subintent['responses'][0]

        return matched_intent['responses'][0]

    def handle_organizational_structure(self, subintent):
        if subintent == "find_manager":
            return "Please provide your name so I can look up your manager."
        elif subintent == "find_hr":
            return "The HR Manager is Beulah Baki. Would you like their contact details?"
        elif subintent == "department_heads":
            return '''Here are the department heads:
- IT: Chandra Mohan Rowthu
- HR: Beulah Baki
- Finance: Supamudu Arrasetty'''
        elif subintent == "reporting_lines":
            return "The reporting lines are based on the org chart. Please provide a name to trace the hierarchy."
        return "I'm not sure which part of the organizational structure you're referring to."

    def handle_attendance_queries(self, subintent):
        if subintent == "mark_attendance":
            return "Your attendance has been marked successfully."
        elif subintent == "view_attendance":
            return "Here is your attendance report for the selected period."
        return "What would you like to do with your attendance?"

    def handle_auto_invoice_queries(self, subintent):
        if subintent == "generate_invoice":
            return "Please provide the customer ID and product details to generate the invoice."
        elif subintent == "view_invoices":
            return "Here are your recent invoices."
        return "Would you like to generate a new invoice or see existing ones?"

    def handle_payroll_queries(self, subintent):
        if subintent == "salary_components":
            return "Your salary includes components like Basic, HRA, LTA, Special Allowance, and Group Medical deductions."
        elif subintent == "component_taxable_status":
            return "Group Medical, LTA, and Special Allowance are taxable. HRA and Other Allowance are non-taxable."
        elif subintent == "salary_status":
            return "Your salary for this month has been processed. You should see it reflected in your account."
        elif subintent == "attendance_impact":
            return "Yes, some components are dependent on attendance. Special Allowance, LTA, and Group Medical vary based on days worked."
        elif subintent == "zero_value_removal":
            return "Components like Special Allowance and LTA are removed if their value is zero."
        elif subintent == "view_payslip":
            return "You can view or download your payslip from the Payroll → Run Payrolls section or request it to be emailed."
        return "Please specify which payroll information you are looking for."

    def handle_expense_queries(self, subintent):
        if subintent == "submit_expense":
            return "Sure, please provide the expense amount and category."
        elif subintent == "view_expense":
            return "Here is a list of your submitted expenses."
        elif subintent == "expense_status":
            return "Let me check the status of your recent expense claims."
        return "What would you like to do with your expenses? You can submit, view, or check status."

    def handle_travel_queries(self, subintent):
        if subintent == "submit_travel_request":
            return "Please provide your travel purpose, destination, and advance amount to submit your travel request."
        elif subintent == "view_travel_requests":
            return "Here are your travel requests with IDs, purpose, and advance details."
        elif subintent == "travel_advance_status":
            return "Your advance request for the trip has been saved. Please follow up with the approving authority."
        elif subintent == "trip_status":
            return "Your trips are currently in 'Saved' status. You will be notified once they are approved."
        elif subintent == "travel_purpose_check":
            return "Most of your trips are listed as Business travel. For specific trip purposes, provide the Travel ID."
        return "Could you please clarify what travel-related info you're looking for?"

    def handle_leave_management(self, subintent):
        if subintent == "apply_leave":
            return "To apply for leave, please provide the leave type (annual, sick, personal), start date, end date, and reason."
        elif subintent == "leave_balance":
            return "Your current leave balance:\n- Annual Leave: 15 days\n- Sick Leave: 10 days\n- Personal Leave: 5 days"
        elif subintent == "leave_status":
            return "Your recent leave requests:\n- Annual Leave (Jan 10-15, 2024): Approved\n- Sick Leave (Feb 5, 2024): Pending approval"
        elif subintent == "cancel_leave":
            return "Which leave application would you like to cancel? Please provide the leave ID or date."
        elif subintent == "leave_history":
            return "Your leave history for the past 6 months:\n- Annual Leave: 5 days (Jan 10-15, 2024)\n- Sick Leave: 2 days (Feb 5-6, 2024)\n- Personal Leave: 1 day (Mar 20, 2024)"
        elif subintent == "team_leaves":
            return "Current team members on leave:\n- John Smith: Annual Leave (Jul 10-15, 2024)\n- Sarah Brown: Sick Leave (Jul 8, 2024)\n- Mike Johnson: Personal Leave (Jul 12, 2024)"
        return "What would you like to do regarding your leaves? You can apply for leave, check your balance, or view your leave history."

    def get_fallback_response(self, user_message):
        query_embedding = self.vectorizer.transform([user_message.lower()])
        faq_embeddings = self.vectorizer.transform(self.faqs['question'].str.lower())
        similarities = cosine_similarity(query_embedding, faq_embeddings)
        max_idx = np.argmax(similarities)
        if similarities[0][max_idx] > 0.5:
            return self.faqs.iloc[max_idx]['answer']
        return "I'm not sure I understand. Could you please rephrase your question?"

    def log_conversation(self, user_id, message, response, intent_data, sentiment):
        conn = sqlite3.connect('erp_chatbot.db')
        c = conn.cursor()
        c.execute('''INSERT INTO conversations
                     (user_id, message, response, intent, subintent, sentiment, confidence, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, message, response,
                   intent_data.get('intent'), intent_data.get('subintent'),
                   sentiment.get('label'), intent_data.get('confidence', 0),
                   datetime.now()))
        conn.commit()
        conn.close()


chatbot = ERPChatbot()


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id', 'anonymous')
    message = data.get('message', '')
    sentiment = chatbot.analyze_sentiment(message)
    intent_data = chatbot.classify_intent(message)
    response = chatbot.generate_response(intent_data, message)
    chatbot.log_conversation(user_id, message, response, intent_data, sentiment)
    return jsonify({
        'response': response,
        'intent': intent_data['intent'],
        'subintent': intent_data.get('subintent'),
        'confidence': intent_data['confidence'],
        'sentiment': sentiment['label'],
        'sentiment_score': sentiment['score']
    })


@app.route('/api/faqs', methods=['GET'])
def get_faqs():
    faqs = chatbot.faqs.to_dict('records')
    return jsonify(faqs)


if __name__ == '__main__':
    app.run(debug=True)