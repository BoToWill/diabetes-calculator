import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import re
import time
import hashlib
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np

class DiabetesCalculator:
    """Advanced AI-powered diabetes calculator with comprehensive analytics."""
    
    def __init__(self):
        self.data_file = "diabetes_data.json"
        self.backup_file = "diabetes_backup.json"
        self.cache_timeout = 300  # 5 minutes
        self.init_session_state()
        self.load_saved_data()
        self.setup_page()
        self.init_performance_monitoring()
    
    def init_session_state(self) -> None:
        """Initialize session state with comprehensive defaults and caching."""
        defaults = {
            'meal_data': [],
            'bu_weight': 12,
            'daily_totals': {},
            'product_history': [],
            'user_profile': {
                'name': '',
                'age': 30,
                'weight': 70,
                'height': 170,
                'activity_level': 'medium',
                'insulin_type': 'rapid',
                'target_glucose': {'min': 4.0, 'max': 7.0}
            },
            'analytics_cache': {},
            'last_cache_update': 0,
            'performance_metrics': {'load_time': 0, 'calculation_time': 0},
            'theme': 'light',
            'language': 'uk',
            'notifications': {'enabled': True, 'reminders': []},
            'ai_suggestions': [],
            'glucose_logs': [],
            'meal_patterns': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def init_performance_monitoring(self) -> None:
        """Initialize performance monitoring and caching."""
        st.session_state.performance_metrics['load_time'] = time.time()
        
    def load_product_database(self) -> Dict[str, List[Dict]]:
        """Load product database from CSV file."""
        products = {}
        
        try:
            with open('table.csv', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines[2:]:  # Skip header and empty line
                line = line.strip()
                if not line or '|' not in line:
                    continue
                    
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 5:
                    category = parts[0]
                    name = parts[1]
                    calories = float(parts[2])
                    protein = float(parts[3])
                    carbs = float(parts[4])
                    
                    if category not in products:
                        products[category] = []
                    
                    products[category].append({
                        'name': name,
                        'calories': calories,
                        'protein': protein,
                        'carbs': carbs
                    })
        except Exception as e:
            st.error(f"Помилка завантаження бази продуктів: {e}")
            
        return products
    
    def load_saved_data(self) -> None:
        """Load saved data with backup and integrity checks."""
        start_time = time.time()
        try:
            # Try primary file first
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    
                # Validate data integrity
                if self.validate_data_integrity(saved_data):
                    self.load_data_to_session(saved_data)
                    st.success("📂 Дані завантажено з попереднього сеансу")
                else:
                    st.warning("⚠️ Основний файл пошкоджено, спроба відновлення з резерву")
                    self.try_load_backup()
            else:
                self.create_backup()
                
        except Exception as e:
            st.error(f"Помилка завантаження даних: {e}")
            self.try_load_backup()
        finally:
            st.session_state.performance_metrics['load_time'] = time.time() - start_time
    
    def validate_data_integrity(self, data: Dict) -> bool:
        """Validate data structure and integrity."""
        required_keys = ['meal_data', 'bu_weight', 'daily_totals', 'product_history']
        return all(key in data for key in required_keys)
    
    def try_load_backup(self) -> None:
        """Try to load data from backup file."""
        if os.path.exists(self.backup_file):
            try:
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                    self.load_data_to_session(backup_data)
                    st.success("✅ Дані відновлено з резервної копії")
            except Exception as e:
                st.error(f"Не вдалося відновити дані: {e}")
    
    def load_data_to_session(self, data: Dict) -> None:
        """Load validated data into session state."""
        if 'meal_data' in data:
            st.session_state.meal_data = data['meal_data']
        if 'bu_weight' in data:
            st.session_state.bu_weight = data['bu_weight']
        if 'daily_totals' in data:
            st.session_state.daily_totals = data['daily_totals']
        if 'product_history' in data:
            st.session_state.product_history = data['product_history']
        if 'user_profile' in data:
            st.session_state.user_profile.update(data['user_profile'])
        if 'glucose_logs' in data:
            st.session_state.glucose_logs = data['glucose_logs']
        if 'meal_patterns' in data:
            st.session_state.meal_patterns = data['meal_patterns']
    
    def save_data_to_file(self) -> None:
        """Save current data with automatic backup and compression."""
        start_time = time.time()
        try:
            data_to_save = {
                'meal_data': st.session_state.meal_data,
                'bu_weight': st.session_state.bu_weight,
                'daily_totals': st.session_state.daily_totals,
                'product_history': st.session_state.product_history,
                'user_profile': st.session_state.user_profile,
                'glucose_logs': st.session_state.glucose_logs,
                'meal_patterns': st.session_state.meal_patterns,
                'last_saved': datetime.now().isoformat(),
                'version': '3.1'
            }
            
            # Create backup before saving
            self.create_backup()
            
            # Save main file
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                
            # Update cache timestamp
            st.session_state.last_cache_update = time.time()
                
        except Exception as e:
            st.error(f"Помилка збереження даних: {e}")
        finally:
            st.session_state.performance_metrics['save_time'] = time.time() - start_time
    
    def create_backup(self) -> None:
        """Create backup of current data."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as src:
                    with open(self.backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
        except Exception as e:
            st.warning(f"Не вдалося створити резервну копію: {e}")
    
    def setup_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Мій Щоденник Діабету",
            page_icon="🍎",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def get_custom_css(self) -> str:
        """Return minimalist CSS with clean design."""
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * {{ font-family: 'Inter', sans-serif; }}
        
        .main-header {{
            font-size: 2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
        }}
        
        .result-card {{
            background: #ffffff;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }}
        
        .result-card h2 {{
            color: #1f2937;
            font-size: 2rem;
            margin: 0;
        }}
        
        .metric-card {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        
        .stButton>button {{
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton>button:hover {{
            background: #2563eb;
        }}
        
        .data-table {{
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .sidebar-section {{
            background: #f9fafb;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid #e5e7eb;
        }}
        
        .ai-suggestion {{
            background: #f0f9ff;
            border-left: 3px solid #3b82f6;
            border-radius: 4px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }}
        
        .glucose-indicator {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 0.25rem;
        }}
        
        .glucose-normal {{ background: #dcfce7; color: #166534; }}
        .glucose-warning {{ background: #fef3c7; color: #92400e; }}
        .glucose-danger {{ background: #fee2e2; color: #991b1b; }}
        
        .performance-metric {{
            background: #f9fafb;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem;
            display: inline-block;
            font-size: 0.8rem;
            border: 1px solid #e5e7eb;
        }}
        </style>
        """
    
    def validate_input(self, product_name: str, carbs_per_100: float, weight: float) -> tuple[bool, str]:
        """Validate user input with comprehensive checks."""
        if not product_name or not product_name.strip():
            return False, "Введіть назву продукту"
        
        if len(product_name.strip()) < 2:
            return False, "Назва занадто коротка"
        
        if carbs_per_100 < 0 or carbs_per_100 > 100:
            return False, "Кількість вуглеводів повинна бути від 0 до 100г"
        
        if weight <= 0 or weight > 10000:
            return False, "Вага повинна бути від 1г до 10кг"
        
        return True, ""
    
    def calculate_bread_units(self, carbs_per_100: float, weight: float, bu_weight: float) -> Dict[str, float]:
        """Enhanced bread units calculation with nutritional analysis."""
        start_time = time.time()
        
        total_carbs = (carbs_per_100 * weight) / 100
        bread_units = total_carbs / bu_weight
        
        # Calculate additional metrics
        calories = self.calculate_calories(carbs_per_100, weight)
        glycemic_load = self.calculate_glycemic_load(carbs_per_100, weight)
        
        result = {
            'total_carbs': round(total_carbs, 1),
            'bread_units': round(bread_units, 2),
            'calories': round(calories, 0),
            'glycemic_load': round(glycemic_load, 1)
        }
        
        st.session_state.performance_metrics['calculation_time'] = time.time() - start_time
        return result
    
    def calculate_calories(self, carbs_per_100: float, weight: float) -> float:
        """Calculate calories from carbohydrates."""
        return (carbs_per_100 * weight) / 100 * 4  # 4 calories per gram of carbs
    
    def calculate_glycemic_load(self, carbs_per_100: float, weight: float, gi: int = 50) -> float:
        """Calculate glycemic load."""
        total_carbs = (carbs_per_100 * weight) / 100
        return (total_carbs * gi) / 100
    
    def get_ai_suggestions(self, current_meal: List[Dict]) -> List[str]:
        """Generate AI-powered meal suggestions."""
        suggestions = []
        
        if not current_meal:
            return suggestions
        
        total_bu = sum(item['ХО'] for item in current_meal)
        total_carbs = sum(item['Вугл. (г)'] for item in current_meal)
        
        # Analyze meal composition
        if total_bu > 5:
            suggestions.append("💡 Високий вміст ХО - розгляньте додавання білків для кращого контролю глюкози")
        
        if total_carbs > 60:
            suggestions.append("⚠️ Велика кількість вуглеводів - рекомендується розподілити на кілька прийомів")
        
        # Check for balanced nutrition
        protein_items = [item for item in current_meal if item['Вугл. (г)'] < 5]
        if len(protein_items) == 0:
            suggestions.append("🥩 Додайте білкові продукти для збалансованого харчування")
        
        # Time-based suggestions
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10 and total_bu < 2:
            suggestions.append("🌅 Сніданок замалий - додайте складні вуглеводи для енергії")
        elif 18 <= current_hour <= 22 and total_bu > 4:
            suggestions.append("🌙 Вечеря занадто важка - зменшіть вуглеводи для кращого сну")
        
        return suggestions
    
    def predict_glucose_impact(self, meal_data: List[Dict]) -> Dict[str, Any]:
        """Predict glucose impact based on meal composition."""
        if not meal_data:
            return {'prediction': 'neutral', 'impact': 0, 'time_to_peak': 60}
        
        total_carbs = sum(item['Вугл. (г)'] for item in meal_data)
        total_bu = sum(item['ХО'] for item in meal_data)
        
        # Simple prediction model
        glucose_rise = total_carbs * 0.3  # Approximate glucose rise
        time_to_peak = 60 + (total_bu * 10)  # Time to peak glucose
        
        if glucose_rise < 50:
            prediction = 'low'
        elif glucose_rise < 100:
            prediction = 'moderate'
        else:
            prediction = 'high'
        
        return {
            'prediction': prediction,
            'impact': round(glucose_rise, 1),
            'time_to_peak': round(time_to_peak, 0),
            'recommendation': self.get_glucose_recommendation(prediction)
        }
    
    def get_glucose_recommendation(self, prediction: str) -> str:
        """Get recommendation based on glucose prediction."""
        recommendations = {
            'low': '✅ Низький ризик - моніторте глюкозу через 1 годину',
            'moderate': '⚠️ Помірний ризик - розгляньте невелику дозу інсуліну',
            'high': '🚨 Високий ризик - рекомендується корекція дозування інсуліну'
        }
        return recommendations.get(prediction, '📊 Моніторте рівень глюкози')
    
    def add_product(self, product_name: str, carbs_per_100: float, weight: float) -> bool:
        """Enhanced product addition with auto-suggestions and database lookup."""
        is_valid, error_msg = self.validate_input(product_name, carbs_per_100, weight)
        
        if not is_valid:
            st.error(error_msg)
            return False
        
        # Check if product exists in database for auto-completion
        product_db = self.load_product_database()
        
        # Find product in CSV database
        product_info = None
        for category, products in product_db.items():
            for product in products:
                if product['name'].lower() == product_name.lower().strip():
                    product_info = product
                    break
            if product_info:
                break
        
        if product_info:
            # Use database values if user input seems incorrect
            if abs(carbs_per_100 - product_info['carbs']) > 5:
                st.info(f"💡 У базі даних: {product_info['carbs']}г вуглеводів на 100г")
        
        calculation = self.calculate_bread_units(carbs_per_100, weight, st.session_state.bu_weight)
        
        product_entry = {
            "Продукт": product_name.strip(),
            "Вага (г)": weight,
            "Вугл. (г)": calculation['total_carbs'],
            "ХО": calculation['bread_units'],
            "Калорії": calculation['calories'],
            "Глікемічне навантаження": calculation['glycemic_load'],
            "Час": datetime.now().strftime("%H:%M"),
            "Дата": datetime.now().strftime("%Y-%m-%d")
        }
        
        st.session_state.meal_data.append(product_entry)
        
        # Add to history with frequency tracking
        self.update_product_history(product_name.strip())
        
        # Update meal patterns
        self.update_meal_patterns(product_entry)
        
        # Auto-save after adding product
        self.save_data_to_file()
        
        # Generate AI suggestions
        ai_suggestions = self.get_ai_suggestions(st.session_state.meal_data)
        if ai_suggestions:
            for suggestion in ai_suggestions[:2]:  # Show top 2 suggestions
                st.info(suggestion)
        
        st.success(f"✅ Додано: {product_name.strip()} ({calculation['bread_units']} ХО, {calculation['calories']} ккал)")
        return True
    
    def update_product_history(self, product_name: str) -> None:
        """Update product history with frequency tracking."""
        if product_name not in st.session_state.product_history:
            st.session_state.product_history.append(product_name)
        
        # Track frequency for smart suggestions
        if 'product_frequency' not in st.session_state:
            st.session_state.product_frequency = {}
        
        st.session_state.product_frequency[product_name] = st.session_state.product_frequency.get(product_name, 0) + 1
    
    def update_meal_patterns(self, product_entry: Dict) -> None:
        """Update meal pattern analysis."""
        current_hour = datetime.now().hour
        meal_type = self.get_meal_type(current_hour)
        
        if meal_type not in st.session_state.meal_patterns:
            st.session_state.meal_patterns[meal_type] = []
        
        st.session_state.meal_patterns[meal_type].append({
            'product': product_entry['Продукт'],
            'bu': product_entry['ХО'],
            'time': current_hour
        })
    
    def get_meal_type(self, hour: int) -> str:
        """Determine meal type based on hour."""
        if 5 <= hour < 11:
            return 'breakfast'
        elif 11 <= hour < 15:
            return 'lunch'
        elif 15 <= hour < 18:
            return 'snack'
        else:
            return 'dinner'
    
    def calculate_totals(self) -> Dict[str, float]:
        """Enhanced total calculation with additional metrics."""
        if not st.session_state.meal_data:
            return {
                'total_carbs': 0, 
                'total_bu': 0, 
                'total_calories': 0, 
                'total_glycemic_load': 0,
                'average_gi': 0
            }
        
        total_carbs = sum(item["Вугл. (г)"] for item in st.session_state.meal_data)
        total_bu = sum(item["ХО"] for item in st.session_state.meal_data)
        total_calories = sum(item.get("Калорії", 0) for item in st.session_state.meal_data)
        total_glycemic_load = sum(item.get("Глікемічне навантаження", 0) for item in st.session_state.meal_data)
        
        return {
            'total_carbs': round(total_carbs, 1),
            'total_bu': round(total_bu, 2),
            'total_calories': round(total_calories, 0),
            'total_glycemic_load': round(total_glycemic_load, 1),
            'average_gi': round(total_glycemic_load / max(total_carbs, 1) * 100, 0) if total_carbs > 0 else 0
        }
    
    def render_analytics_dashboard(self) -> None:
        """Render comprehensive analytics dashboard."""
        st.markdown("### 📊 Аналітика та статистика")
        
        # Check if we have enough data
        if not st.session_state.daily_totals:
            st.info("📈 Недостатньо даних для аналітики. Додайте прийоми їжі для побудови графіків.")
            return
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Графіки", "🍽️ Патерни", "🎯 Цілі", "📅 Звіти"])
        
        with tab1:
            self.render_charts_tab()
        
        with tab2:
            self.render_patterns_tab()
        
        with tab3:
            self.render_goals_tab()
        
        with tab4:
            self.render_reports_tab()
    
    def render_charts_tab(self) -> None:
        """Render charts and visualizations."""
        # Prepare data for visualization
        daily_data = self.prepare_daily_data()
        
        if not daily_data:
            st.warning("📊 Немає даних для відображення")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily BU trend
            fig_bu = go.Figure()
            fig_bu.add_trace(go.Scatter(
                x=list(daily_data.keys()),
                y=[d['total_bu'] for d in daily_data.values()],
                mode='lines+markers',
                name='ХО на день',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8)
            ))
            fig_bu.update_layout(
                title='📈 Динаміка Хлібних Одиниць',
                xaxis_title='Дата',
                yaxis_title='ХО',
                height=400
            )
            st.plotly_chart(fig_bu, use_container_width=True)
        
        with col2:
            # Carbs distribution
            fig_carbs = go.Figure()
            fig_carbs.add_trace(go.Bar(
                x=list(daily_data.keys()),
                y=[d['total_carbs'] for d in daily_data.values()],
                name='Вуглеводи (г)',
                marker_color='#8b5cf6'
            ))
            fig_carbs.update_layout(
                title='🍞 Вуглеводи по днях',
                xaxis_title='Дата',
                yaxis_title='Вуглеводи (г)',
                height=400
            )
            st.plotly_chart(fig_carbs, use_container_width=True)
        
        # Meal distribution pie chart
        if st.session_state.meal_patterns:
            meal_counts = {meal_type: len(meals) for meal_type, meals in st.session_state.meal_patterns.items()}
            
            fig_pie = go.Figure()
            fig_pie.add_trace(go.Pie(
                labels=list(meal_counts.keys()),
                values=list(meal_counts.values()),
                name="Розподіл прийомів"
            ))
            fig_pie.update_layout(
                title='🍽️ Розподіл прийомів їжі',
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_patterns_tab(self) -> None:
        """Render meal patterns analysis."""
        if not st.session_state.meal_patterns:
            st.info("🔍 Аналіз патернів буде доступний після накопичення даних")
            return
        
        st.markdown("#### 🕐 Аналіз часу прийомів")
        
        for meal_type, meals in st.session_state.meal_patterns.items():
            if meals:
                avg_bu = sum(m['bu'] for m in meals) / len(meals)
                common_time = max(set(m['time'] for m in meals), key=lambda x: sum(1 for m in meals if m['time'] == x))
                
                meal_names = [m['product'] for m in meals]
                most_common = max(set(meal_names), key=meal_names.count) if meal_names else 'N/A'
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{meal_type.title()}</h4>
                    <p><strong>Середньо ХО:</strong> {avg_bu:.1f}</p>
                    <p><strong>Найчастіший час:</strong> {common_time}:00</p>
                    <p><strong>Популярний продукт:</strong> {most_common}</p>
                    <p><strong>Кількість прийомів:</strong> {len(meals)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_goals_tab(self) -> None:
        """Render goals and targets tracking."""
        st.markdown("#### 🎯 Ваї цілі та досягнення")
        
        # Goal setting
        with st.expander("⚙️ Налаштування цілей"):
            daily_bu_goal = st.number_input(
                "Денна ціль по ХО:", 
                min_value=5, 
                max_value=20, 
                value=10, 
                step=0.5
            )
            
            glucose_target_min = st.number_input(
                "Цільовий рівень глюкози (мін):", 
                min_value=3.0, 
                max_value=10.0, 
                value=4.0, 
                step=0.1
            )
            
            glucose_target_max = st.number_input(
                "Цільовий рівень глюкози (макс):", 
                min_value=5.0, 
                max_value=15.0, 
                value=7.0, 
                step=0.1
            )
        
        # Progress tracking
        today_totals = self.calculate_totals()
        progress_percentage = (today_totals['total_bu'] / daily_bu_goal) * 100 if daily_bu_goal > 0 else 0
        
        st.markdown(f"""
        <div class="result-card">
            <h3>Сьогоднішній прогрес</h3>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 10px; margin: 10px 0;">
                <div style="background: linear-gradient(90deg, #10b981 {min(progress_percentage, 100)}%, rgba(255,255,255,0.3) {min(progress_percentage, 100)}%); border-radius: 8px; padding: 15px; text-align: center;">
                    <strong>{progress_percentage:.1f}%</strong> від деннї цілі
                </div>
            </div>
            <p>{today_totals['total_bu']} / {daily_bu_goal} ХО</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_reports_tab(self) -> None:
        """Generate and display reports."""
        st.markdown("#### 📄 Звіти та експорт")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Звіт за тиждень", use_container_width=True):
                self.generate_weekly_report()
        
        with col2:
            if st.button("📅 Звіт за місяць", use_container_width=True):
                self.generate_monthly_report()
        
        # Export options
        st.markdown("##### 📤 Експорт даних")
        
        export_format = st.selectbox(
            "Формат експорту:",
            ["CSV", "JSON", "PDF"]
        )
        
        if st.button("📥 Завантажити звіт", use_container_width=True):
            self.export_data(export_format)
    
    def prepare_daily_data(self) -> Dict[str, Dict]:
        """Prepare daily data for visualization."""
        daily_data = {}
        
        for date, meals in st.session_state.daily_totals.items():
            total_bu = sum(meal['totals']['total_bu'] for meal in meals)
            total_carbs = sum(meal['totals']['total_carbs'] for meal in meals)
            
            daily_data[date] = {
                'total_bu': total_bu,
                'total_carbs': total_carbs,
                'meal_count': len(meals)
            }
        
        return daily_data
    
    def generate_weekly_report(self) -> None:
        """Generate weekly summary report."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Filter data for the week
        weekly_data = {
            date: data for date, data in st.session_state.daily_totals.items()
            if start_date <= datetime.fromisoformat(date) <= end_date
        }
        
        if not weekly_data:
            st.warning("📊 Немає даних за останній тиждень")
            return
        
        # Calculate weekly statistics
        total_meals = sum(len(meals) for meals in weekly_data.values())
        avg_bu_per_day = sum(
            sum(meal['totals']['total_bu'] for meal in meals) 
            for meals in weekly_data.values()
        ) / len(weekly_data)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Звіт за тиждень ({start_date.strftime('%d.%m')} - {end_date.strftime('%d.%m')})</h4>
            <p><strong>Загальна кількість прийомів:</strong> {total_meals}</p>
            <p><strong>Середньо ХО на день:</strong> {avg_bu_per_day:.1f}</p>
            <p><strong>Кількість днів з даними:</strong> {len(weekly_data)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def generate_monthly_report(self) -> None:
        """Generate monthly summary report."""
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        monthly_data = {
            date: data for date, data in st.session_state.daily_totals.items()
            if datetime.fromisoformat(date).month == current_month and 
               datetime.fromisoformat(date).year == current_year
        }
        
        if not monthly_data:
            st.warning("📊 Немає даних за поточний місяць")
            return
        
        total_bu = sum(
            sum(meal['totals']['total_bu'] for meal in meals) 
            for meals in monthly_data.values()
        )
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📅 Звіт за місяць</h4>
            <p><strong>Загально ХО за місяць:</strong> {total_bu:.1f}</p>
            <p><strong>Середньо ХО на день:</strong> {total_bu / len(monthly_data):.1f}</p>
            <p><strong>Кількість днів з даними:</strong> {len(monthly_data)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def export_data(self, format_type: str) -> None:
        """Export data in specified format."""
        if format_type == "CSV":
            self.export_to_csv()
        elif format_type == "JSON":
            self.export_to_json()
        elif format_type == "PDF":
            st.info("📄 Експорт в PDF буде доступний в наступній версії")
    
    def export_to_json(self) -> None:
        """Export data to JSON format."""
        export_data = {
            'user_profile': st.session_state.user_profile,
            'daily_totals': st.session_state.daily_totals,
            'meal_patterns': st.session_state.meal_patterns,
            'export_date': datetime.now().isoformat()
        }
        
        json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="📥 Завантажити JSON",
            data=json_data,
            file_name=f"diabetes_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
    
    def render_sidebar(self) -> None:
        """Render enhanced sidebar with comprehensive settings."""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("⚙️ Налаштування")
            
            # User Profile Section
            with st.expander("👤 Профіль користувача"):
                name = st.text_input(
                    "Ім'я:", 
                    value=st.session_state.user_profile['name']
                )
                age = st.number_input(
                    "Вік:", 
                    min_value=1, 
                    max_value=120, 
                    value=st.session_state.user_profile['age']
                )
                weight = st.number_input(
                    "Вага (кг):", 
                    min_value=20, 
                    max_value=200, 
                    value=st.session_state.user_profile['weight']
                )
                height = st.number_input(
                    "Зріст (см):", 
                    min_value=100, 
                    max_value=250, 
                    value=st.session_state.user_profile['height']
                )
                
                activity_level = st.selectbox(
                    "Рівень активності:",
                    ['low', 'medium', 'high'],
                    index=['low', 'medium', 'high'].index(st.session_state.user_profile['activity_level'])
                )
                
                insulin_type = st.selectbox(
                    "Тип інсуліну:",
                    ['rapid', 'short', 'intermediate', 'long'],
                    index=['rapid', 'short', 'intermediate', 'long'].index(st.session_state.user_profile['insulin_type'])
                )
                
                # Update profile if changed
                if st.button("💾 Оновити профіль"):
                    st.session_state.user_profile.update({
                        'name': name,
                        'age': age,
                        'weight': weight,
                        'height': height,
                        'activity_level': activity_level,
                        'insulin_type': insulin_type
                    })
                    self.save_data_to_file()
                    st.success("✅ Профіль оновлено!")
            
            # BU Weight Configuration
            st.markdown("#### 📊 Хлібні одиниці")
            bu_weight = st.number_input(
                "Вуглеводів в 1 ХО (грам):", 
                min_value=8, 
                max_value=15, 
                value=st.session_state.bu_weight, 
                step=1,
                help="Зазвичай 1 ХО = 10-12 г вуглеводів"
            )
            
            if bu_weight != st.session_state.bu_weight:
                st.session_state.bu_weight = bu_weight
                self.save_data_to_file()
                st.rerun()
            
            st.info(f"📊 Поточне значення: **1 ХО = {bu_weight} г**")
            
            # Glucose Tracking
            st.markdown("#### 🩸 Глюкоза")
            with st.expander("Додати вимірювання"):
                glucose_level = st.number_input(
                    "Рівень глюкози (ммоль/л):", 
                    min_value=1.0, 
                    max_value=30.0, 
                    step=0.1
                )
                glucose_time = st.selectbox(
                    "Час вимірювання:",
                    ["Перед їжею", "Після їжі", "Вранці", "Перед сном"]
                )
                
                if st.button("📝 Додати вимірювання"):
                    glucose_entry = {
                        'level': glucose_level,
                        'time': glucose_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.glucose_logs.append(glucose_entry)
                    self.save_data_to_file()
                    st.success("✅ Вимірювання додано!")
            
            # Recent glucose readings
            if st.session_state.glucose_logs:
                st.markdown("**Останні вимірювання:**")
                recent_glucose = st.session_state.glucose_logs[-3:]
                for entry in reversed(recent_glucose):
                    level = entry['level']
                    status = self.get_glucose_status(level)
                    st.markdown(f"<span class='glucose-indicator glucose-{status}'>{level} ммоль/л</span> {entry['time']}", unsafe_allow_html=True)
            
            # Action Buttons
            st.markdown("#### 🛠️ Дії")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Очистити", use_container_width=True):
                    st.session_state.meal_data = []
                    self.save_data_to_file()
                    st.rerun()
            
            with col2:
                if st.button("💾 Зберегти", use_container_width=True):
                    self.save_meal_data()
            
            # Statistics
            if st.session_state.meal_data:
                totals = self.calculate_totals()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📈 Статистика прийому</h4>
                    <p><strong>Продуктів:</strong> {len(st.session_state.meal_data)}</p>
                    <p><strong>Всього ХО:</strong> {totals['total_bu']}</p>
                    <p><strong>Всього вугл:</strong> {totals['total_carbs']}г</p>
                    <p><strong>Калорії:</strong> {totals['total_calories']} ккал</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance Metrics
            if st.session_state.performance_metrics:
                st.markdown("#### ⚡ Продуктивність")
                metrics = st.session_state.performance_metrics
                st.markdown(f"""
                <div class="performance-metric">⏱️ Завантаження: {metrics.get('load_time', 0):.3f}с</div>
                <div class="performance-metric">🧮 Розрахунок: {metrics.get('calculation_time', 0):.3f}с</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def get_glucose_status(self, level: float) -> str:
        """Determine glucose status based on level."""
        target_min = st.session_state.user_profile['target_glucose']['min']
        target_max = st.session_state.user_profile['target_glucose']['max']
        
        if target_min <= level <= target_max:
            return 'normal'
        elif level < target_min:
            return 'danger'
        else:
            return 'warning'
    
    def render_main_interface(self) -> None:
        """Render minimalist main application interface."""
        st.markdown('<h1 class="main-header">🍎 Калькулятор Хлібних Одиниць</h1>', unsafe_allow_html=True)
        st.markdown("**Простий та точний підрахунок вуглеводів**")
        
        # Simple navigation
        tab1, tab2, tab3 = st.tabs(["🍽️ Прийом їжі", "📊 Аналітика", "🧮 Інсулін"])
        
        with tab1:
            self.render_meal_input_tab()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_insulin_calculator_tab()
    
    def render_meal_input_tab(self) -> None:
        """Render meal input and tracking tab."""
        # Product Input Form
        st.markdown("### 📝 Додати продукт")
        
        # Smart auto-complete with frequency
        product_suggestions = self.get_smart_suggestions()
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
        
        with col1:
            product_name = st.text_input(
                "Назва продукту", 
                placeholder="Наприклад: Яблуко",
                help="Почніть вводити для автозаповнення"
            )
            
            # Smart suggestions based on frequency and time
            if product_name and len(product_name) > 1:
                suggestions = [p for p in product_suggestions if product_name.lower() in p.lower()]
                if suggestions:
                    selected = st.selectbox("💡 Пропозиції:", suggestions, index=0)
                    if selected:
                        product_name = selected
                        # Auto-fill carbs if product is in database
                        product_db = self.get_product_database()
                        if selected.lower() in product_db:
                            st.info(f"📊 Вуглеводи: {product_db[selected.lower()]['carbs_per_100']}г на 100г")
        
        with col2:
            carbs_per_100 = st.number_input(
                "Вуглеводи на 100г", 
                min_value=0.0, 
                max_value=100.0,
                step=0.1, 
                format="%.1f",
                help="Згідно з етикеткою продукту"
            )
        
        with col3:
            weight = st.number_input(
                "Вага (г)", 
                min_value=1, 
                max_value=10000,
                step=5,
                help="Фактична вага порції"
            )
        
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Додати", use_container_width=True, type="primary"):
                self.add_product(product_name, carbs_per_100, weight)
        
        # Quick Add Templates with AI recommendations
        self.render_smart_quick_add()
        
        # Meal Data Display
        self.render_enhanced_meal_table()
        
        # Results Display with predictions
        self.render_enhanced_results()
    
    def get_smart_suggestions(self) -> List[str]:
        """Get smart product suggestions based on frequency and time."""
        current_hour = datetime.now().hour
        meal_type = self.get_meal_type(current_hour)
        
        # Get frequency-based suggestions
        if 'product_frequency' in st.session_state:
            freq_suggestions = sorted(
                st.session_state.product_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            freq_names = [item[0] for item in freq_suggestions]
        else:
            freq_names = []
        
        # Get time-based suggestions
        time_suggestions = []
        if meal_type in st.session_state.meal_patterns:
            recent_meals = st.session_state.meal_patterns[meal_type][-5:]
            time_suggestions = list(set(m['product'] for m in recent_meals))
        
        # Combine and deduplicate
        all_suggestions = list(dict.fromkeys(freq_names + time_suggestions))
        return all_suggestions[:10]
    
    def render_smart_quick_add(self) -> None:
        """Render category-based quick add from CSV database."""
        st.markdown("### ⚡ Швидке додавання")
        
        products = self.load_product_database()
        
        if not products:
            st.warning("База продуктів не завантажена")
            return
        
        # Category selection
        category = st.selectbox(
            "Оберіть категорію:",
            options=list(products.keys()),
            key="quick_add_category"
        )
        
        if category and products[category]:
            st.markdown(f"#### 📋 {category}")
            
            # Display products in a grid
            cols = st.columns(3)
            
            for i, product in enumerate(products[category]):
                with cols[i % 3]:
                    # Product card with nutritional info
                    st.markdown(f"""
                    <div class="metric-card" style="cursor: pointer;">
                        <h5 style="margin: 0; font-size: 0.9rem;">{product['name']}</h5>
                        <p style="margin: 0.25rem 0; font-size: 0.8rem; color: #666;">
                            🍖 {product['protein']}г білка | 🍞 {product['carbs']}г вугл | 🔥 {product['calories']} ккал
                        </p>
                        <p style="margin: 0; font-size: 0.75rem; color: #3b82f6;">
                            ХО: {(product['carbs'] / st.session_state.bu_weight):.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button with default weight
                    if st.button(f"Додати {product['name']}", key=f"quick_{i}"):
                        # Use standard portion size (100g for most products)
                        default_weight = 100
                        if 'шматок' in product['name'].lower() or 'порція' in product['name'].lower():
                            default_weight = 150  # Larger portion for prepared dishes
                        
                        self.add_product(product['name'], product['carbs'], default_weight)
    
    def render_enhanced_meal_table(self) -> None:
        """Render enhanced meal data table with additional features."""
        if not st.session_state.meal_data:
            st.info("👆 Додайте продукти, щоб побачити розрахунок")
            return
        
        st.markdown("### 🍽️ Ваш прийом їжі")
        
        df = pd.DataFrame(st.session_state.meal_data)
        
        # Enhanced table display with new columns
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Продукт": st.column_config.TextColumn("Продукт", width="large"),
                "Вага (г)": st.column_config.NumberColumn("Вага (г)", format="%d г"),
                "Вугл. (г)": st.column_config.NumberColumn("Вуглеводи (г)", format="%.1f г"),
                "ХО": st.column_config.NumberColumn(
                    "ХО", 
                    format="%.2f ⭐",
                    help="Хлібні одиниці"
                ),
                "Калорії": st.column_config.NumberColumn("Калорії", format="%d ккал"),
                "Глікемічне навантаження": st.column_config.NumberColumn("ГН", format="%.1f"),
                "Час": st.column_config.TextColumn("Час", width="small")
            }
        )
        
        # Enhanced actions
        if len(st.session_state.meal_data) > 1:
            st.markdown("#### 🛠️ Дії з продуктами")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                item_to_remove = st.selectbox(
                    "Видалити продукт:",
                    options=range(len(st.session_state.meal_data)),
                    format_func=lambda i: f"{st.session_state.meal_data[i]['Продукт']} ({st.session_state.meal_data[i]['ХО']} ХО)"
                )
                
                if st.button("🗑️ Видалити", key="remove_item"):
                    removed = st.session_state.meal_data.pop(item_to_remove)
                    self.save_data_to_file()
                    st.success(f"Видалено: {removed['Продукт']}")
                    st.rerun()
            
            with col2:
                if st.button("📊 Експорт в CSV", key="export_csv"):
                    self.export_to_csv()
            
            with col3:
                if st.button("🔄 Дублювати прийом", key="duplicate_meal"):
                    # Duplicate current meal for next time
                    duplicated = [item.copy() for item in st.session_state.meal_data]
                    for item in duplicated:
                        item['Час'] = datetime.now().strftime("%H:%M")
                    st.session_state.meal_data.extend(duplicated)
                    self.save_data_to_file()
                    st.success("✅ Прийом продубльовано!")
                    st.rerun()
    
            return
        
        st.markdown("### 🍽️ Ваш прийом їжі")
        
        df = pd.DataFrame(st.session_state.meal_data)
        
        # Enhanced table display
        st.dataframe(
            df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Продукт": st.column_config.TextColumn("Продукт", width="large"),
                "Вага (г)": st.column_config.NumberColumn("Вага (г)", format="%d г"),
                "Вугл. (г)": st.column_config.NumberColumn("Вуглеводи (г)", format="%.1f г"),
                "ХО": st.column_config.NumberColumn(
                    "ХО", 
                    format="%.2f ⭐",
                    help="Хлібні одиниці"
                ),
                "Час": st.column_config.TextColumn("Час", width="small")
            }
        )
        
        # Individual item actions
        if len(st.session_state.meal_data) > 1:
            st.markdown("#### 🛠️ Дії з продуктами")
            col1, col2 = st.columns(2)
            
            with col1:
                item_to_remove = st.selectbox(
                    "Видалити продукт:",
                    options=range(len(st.session_state.meal_data)),
                    format_func=lambda i: f"{st.session_state.meal_data[i]['Продукт']} ({st.session_state.meal_data[i]['ХО']} ХО)"
                )
                
                if st.button("🗑️ Видалити", key="remove_item"):
                    removed = st.session_state.meal_data.pop(item_to_remove)
                    self.save_data_to_file()
                    st.success(f"Видалено: {removed['Продукт']}")
                    st.rerun()
            
            with col2:
                if st.button("📊 Експорт в CSV", key="export_csv"):
                    self.export_to_csv()
    
    def render_enhanced_results(self) -> None:
        """Render clean calculation results."""
        if not st.session_state.meal_data:
            return
        
        totals = self.calculate_totals()
        
        # Clean result card
        st.markdown(f"""
        <div class="result-card">
            <h2>{totals['total_bu']} ХО</h2>
            <p>{totals['total_carbs']} г вуглеводів</p>
            <p>{totals['total_calories']} ккал</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Середнє ХО", 
                f"{totals['total_bu'] / len(st.session_state.meal_data):.2f}"
            )
        
        with col2:
            st.metric(
                "Глікемічне навантаження", 
                f"{totals['total_glycemic_load']:.1f}"
            )
        
        with col3:
            insulin_ratio = st.number_input(
                "Співвідношення інсуліну",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                value=1.0,
                help="Одиниці інсуліну на 1 ХО"
            )
        
        # Insulin calculation
        insulin_needed = totals['total_bu'] * insulin_ratio
        st.markdown(f"""
        <div class="metric-card">
            <h4>💉 Розрахунок інсуліну</h4>
            <p><strong>Потрібно інсуліну:</strong> {insulin_needed:.1f} од.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_insulin_calculator_tab(self) -> None:
        """Render advanced insulin calculator."""
        st.markdown("### 🧮 Калькулятор інсуліну")
        
        # Current glucose
        col1, col2 = st.columns(2)
        
        with col1:
            current_glucose = st.number_input(
                "Поточний рівень глюкози (ммоль/л):",
                min_value=1.0,
                max_value=30.0,
                step=0.1,
                value=5.5
            )
        
        with col2:
            target_glucose = st.number_input(
                "Цільовий рівень глюкози (ммоль/л):",
                min_value=3.0,
                max_value=15.0,
                step=0.1,
                value=6.0
            )
        
        # Insulin sensitivity
        st.markdown("#### 🎯 Чутливість до інсуліну")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            carb_ratio = st.number_input(
                "Співвідношення вуглеводів (г на 1 од. інсуліну):",
                min_value=5,
                max_value=20,
                step=1,
                value=10
            )
        
        with col2:
            correction_factor = st.number_input(
                "Корекційний фактор (ммоль/л на 1 од. інсуліну):",
                min_value=1.0,
                max_value=5.0,
                step=0.1,
                value=2.0
            )
        
        with col3:
            active_insulin = st.number_input(
                "Активний інсулін (од.):",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                value=0.0
            )
        
        # Calculate insulin doses
        if st.button("🧮 Розрахувати дозування", use_container_width=True, type="primary"):
            totals = self.calculate_totals()
            
            # Meal insulin
            meal_insulin = totals['total_carbs'] / carb_ratio
            
            # Correction insulin
            glucose_diff = current_glucose - target_glucose
            correction_insulin = glucose_diff / correction_factor if glucose_diff > 0 else 0
            
            # Total insulin
            total_insulin = meal_insulin + correction_insulin - active_insulin
            
            st.markdown(f"""
            <div class="result-card">
                <h3>💉 Рекомендоване дозування</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <h4>На їжу:</h4>
                        <p style="font-size: 1.5rem; margin: 0;">{meal_insulin:.1f} од.</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                        <h4>На корекцію:</h4>
                        <p style="font-size: 1.5rem; margin: 0;">{correction_insulin:.1f} од.</p>
                    </div>
                </div>
                <p style="font-size: 1.8rem; margin: 1rem 0;"><strong>Разом: {max(0, total_insulin):.1f} од.</strong></p>
                <p style="margin: 0; opacity: 0.9;">Враховуючи активний інсулін: {active_insulin:.1f} од.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_mobile_mode_tab(self) -> None:
        """Render mobile-optimized interface."""
        st.markdown("### 📱 Мобільний режим")
        st.info("📲 Оптимізований інтерфейс для використання на мобільних пристроях")
        
        # Simplified input for mobile
        st.markdown("#### 🚀 Швидкий ввід")
        
        # Quick product selector
        quick_products = {
            "Хліб": (50, 49),
            "Яблуко": (150, 14),
            "Йогурт": (200, 5),
            "Каша": (100, 25),
            "М'ясо": (150, 0),
            "Овочі": (200, 8)
        }
        
        cols = st.columns(2)
        for i, (name, (weight, carbs)) in enumerate(quick_products.items()):
            with cols[i % 2]:
                if st.button(f"🍽️ {name}\n{weight}г", use_container_width=True):
                    self.add_product(name, carbs, weight)
        
        # Current meal summary
        if st.session_state.meal_data:
            totals = self.calculate_totals()
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Поточний прийом</h4>
                <p><strong>ХО:</strong> {totals['total_bu']}</p>
                <p><strong>Вуглеводи:</strong> {totals['total_carbs']}г</p>
                <p><strong>Калорії:</strong> {totals['total_calories']} ккал</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Очистити прийом", use_container_width=True):
                st.session_state.meal_data = []
                self.save_data_to_file()
                st.rerun()
    
    def save_meal_data(self) -> None:
        """Save current meal data to session storage."""
        if st.session_state.meal_data:
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in st.session_state.daily_totals:
                st.session_state.daily_totals[today] = []
            
            meal_entry = {
                'timestamp': datetime.now().isoformat(),
                'data': st.session_state.meal_data.copy(),
                'totals': self.calculate_totals()
            }
            
            st.session_state.daily_totals[today].append(meal_entry)
            self.save_data_to_file()
            st.success("💾 Прийом їжі збережено!")
    
    def export_to_csv(self) -> None:
        """Export meal data to CSV."""
        if st.session_state.meal_data:
            df = pd.DataFrame(st.session_state.meal_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Завантажити CSV",
                data=csv,
                file_name=f"meal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    def render_footer(self) -> None:
        """Render footer with disclaimer and info."""
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **⚠️ Важливо:**
            - Цей калькулятор є допоміжним інструментом
            - Завжди перевіряйте розрахунки
            - Консультуйтеся з ендокринологом
            """)
        
        with col2:
            st.markdown("""
            **📞 Підтримка:**
            - Для питань та пропозицій
            - Версія 2.0 Pro
            - © 2024 Diabetes Calculator
            """)
            
            # Data management info
            if os.path.exists(self.data_file):
                file_size = os.path.getsize(self.data_file)
                st.info(f"📁 Файл даних: {file_size} байт")
            
            # Manual save button
            if st.button("💾 Примусово зберегти", key="manual_save"):
                self.save_data_to_file()
                st.success("✅ Дані збережено!")
    
    def run(self) -> None:
        """Main application entry point with enhanced error handling."""
        try:
            # Apply custom CSS
            st.markdown(self.get_custom_css(), unsafe_allow_html=True)
            
            # Performance monitoring
            start_time = time.time()
            
            # Render components
            self.render_sidebar()
            self.render_main_interface()
            self.render_footer()
            
            # Update performance metrics
            load_time = time.time() - start_time
            st.session_state.performance_metrics['total_load_time'] = load_time
            
            # Show performance info in development
            if st.checkbox("🔧 Показати технічну інформацію", key="show_debug"):
                st.markdown(f"""
                <div class="performance-metric">
                    🚀 Час завантаження: {load_time:.3f}с
                    📊 Розмір даних: {len(str(st.session_state))} символів
                    🔄 Оновлень сесії: {st.session_state.get('rerun_count', 0)}
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"🚨 Помилка програми: {str(e)}")
            st.error("Будь ласка, перезавантажте сторінку або зв'яжіться з підтримкою.")
            
            # Log error for debugging
            if 'error_log' not in st.session_state:
                st.session_state.error_log = []
            
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'session_data_size': len(str(st.session_state))
            }
            st.session_state.error_log.append(error_entry)
    
    def render_footer(self) -> None:
        """Render enhanced footer with comprehensive information."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **⚠️ Важливо:**
            - Цей калькулятор є допоміжним інструментом
            - Завжди перевіряйте розрахунки
            - Консультуйтеся з ендокринологом
            """)
        
        with col2:
            st.markdown("""
            **📞 Підтримка:**
            - Для питань та пропозицій
            - Версія 3.1
            - 2026 Diabetes Calculator
            """)
            
            # Data management info
            if os.path.exists(self.data_file):
                file_size = os.path.getsize(self.data_file)
                st.info(f"📁 Файл даних: {file_size} байт")
        
        with col3:
            st.markdown("**📈 Статистика:**")
            
            # Show usage statistics
            total_meals = len(st.session_state.daily_totals)
            total_products = len(st.session_state.product_history)
            glucose_readings = len(st.session_state.glucose_logs)
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Прийомів їжі:</strong> {total_meals}</p>
                <p><strong>Продуктів в базі:</strong> {total_products}</p>
                <p><strong>Вимірювань глюкози:</strong> {glucose_readings}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Manual save button
            if st.button("💾 Примусово зберегти", key="manual_save"):
                self.save_data_to_file()
                st.success("✅ Дані збережено!")
                st.balloons()
        
        # Version info and updates
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
            🚀 Diabetes Calculator v3.1 | Працює на Streamlit | 
            <a href="#" onclick="alert('Оновлення доступні!')">Перевірити оновлення</a>
        </div>
        """, unsafe_allow_html=True)


# Initialize and run the application
if __name__ == "__main__":
    app = DiabetesCalculator()
    app.run()