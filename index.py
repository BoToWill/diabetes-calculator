import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import re

class DiabetesCalculator:
    """Advanced diabetes calculator with bread units tracking."""
    
    def __init__(self):
        self.data_file = "diabetes_data.json"
        self.init_session_state()
        self.load_saved_data()
        self.setup_page()
    
    def init_session_state(self) -> None:
        """Initialize session state with proper defaults."""
        defaults = {
            'meal_data': [],
            'bu_weight': 12,
            'daily_totals': {},
            'product_history': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_saved_data(self) -> None:
        """Load saved data from JSON file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    
                # Load data into session state
                if 'meal_data' in saved_data:
                    st.session_state.meal_data = saved_data['meal_data']
                if 'bu_weight' in saved_data:
                    st.session_state.bu_weight = saved_data['bu_weight']
                if 'daily_totals' in saved_data:
                    st.session_state.daily_totals = saved_data['daily_totals']
                if 'product_history' in saved_data:
                    st.session_state.product_history = saved_data['product_history']
                    
                st.success("📂 Дані завантажено з попереднього сеансу")
        except Exception as e:
            st.error(f"Помилка завантаження даних: {e}")
    
    def save_data_to_file(self) -> None:
        """Save current data to JSON file."""
        try:
            data_to_save = {
                'meal_data': st.session_state.meal_data,
                'bu_weight': st.session_state.bu_weight,
                'daily_totals': st.session_state.daily_totals,
                'product_history': st.session_state.product_history,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.error(f"Помилка збереження даних: {e}")
    
    def setup_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Мій Щоденник Діабету",
            page_icon="🍎",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def get_custom_css(self) -> str:
        """Return optimized CSS for better UI."""
        return """
        <style>
        .main-header {font-size: 2.5rem; font-weight: 700; color: #1f2937; margin-bottom: 1rem;}
        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            margin: 1.5rem 0;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem 0;
        }
        .stButton>button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(238, 90, 36, 0.4);
        }
        .data-table {border-radius: 10px; overflow: hidden;}
        .sidebar-section {background: #f1f5f9; border-radius: 10px; padding: 1rem; margin: 1rem 0;}
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
        """Calculate bread units and total carbs."""
        total_carbs = (carbs_per_100 * weight) / 100
        bread_units = total_carbs / bu_weight
        
        return {
            'total_carbs': round(total_carbs, 1),
            'bread_units': round(bread_units, 2)
        }
    
    def add_product(self, product_name: str, carbs_per_100: float, weight: float) -> bool:
        """Add product to meal list with validation."""
        is_valid, error_msg = self.validate_input(product_name, carbs_per_100, weight)
        
        if not is_valid:
            st.error(error_msg)
            return False
        
        calculation = self.calculate_bread_units(carbs_per_100, weight, st.session_state.bu_weight)
        
        product_entry = {
            "Продукт": product_name.strip(),
            "Вага (г)": weight,
            "Вугл. (г)": calculation['total_carbs'],
            "ХО": calculation['bread_units'],
            "Час": datetime.now().strftime("%H:%M")
        }
        
        st.session_state.meal_data.append(product_entry)
        
        # Add to history for autocomplete
        if product_name.strip() not in st.session_state.product_history:
            st.session_state.product_history.append(product_name.strip())
        
        # Auto-save after adding product
        self.save_data_to_file()
        
        st.success(f"✅ Додано: {product_name.strip()} ({calculation['bread_units']} ХО)")
        return True
    
    def calculate_totals(self) -> Dict[str, float]:
        """Calculate total carbs and bread units for current meal."""
        if not st.session_state.meal_data:
            return {'total_carbs': 0, 'total_bu': 0}
        
        total_carbs = sum(item["Вугл. (г)"] for item in st.session_state.meal_data)
        total_bu = sum(item["ХО"] for item in st.session_state.meal_data)
        
        return {
            'total_carbs': round(total_carbs, 1),
            'total_bu': round(total_bu, 2)
        }
    
    def render_sidebar(self) -> None:
        """Render enhanced sidebar with settings."""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("⚙️ Налаштування")
            
            # BU Weight Configuration
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
            
            # Action Buttons
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
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_main_interface(self) -> None:
        """Render main application interface."""
        st.markdown('<h1 class="main-header">🍎 Калькулятор Хлібних Одиниць</h1>', unsafe_allow_html=True)
        st.markdown("**Професійний інструмент для точного підрахунку вуглеводів**")
        
        # Product Input Form
        st.markdown("### 📝 Додати продукт")
        
        # Auto-complete for product names
        product_suggestions = st.session_state.product_history[:5] if st.session_state.product_history else []
        
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])
        
        with col1:
            product_name = st.text_input(
                "Назва продукту", 
                placeholder="Наприклад: Яблуко",
                help="Почніть вводити для автозаповнення"
            )
            
            # Show suggestions if available
            if product_name and len(product_name) > 1:
                suggestions = [p for p in product_suggestions if product_name.lower() in p.lower()]
                if suggestions:
                    selected = st.selectbox("💡 Пропозиції:", suggestions, index=0)
                    if selected:
                        product_name = selected
        
        with col2:
            carbs_per_100 = st.number_input(
                "Вуглеводи на 100г", 
                min_value=0.0, 
                max_value=100.0,
                step=0.1, 
                format="%.1f",
                help="Згідно з етикетки продукту"
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
        
        # Quick Add Templates
        self.render_quick_add_templates()
        
        # Meal Data Display
        self.render_meal_table()
        
        # Results Display
        self.render_results()
    
    def render_quick_add_templates(self) -> None:
        """Render quick add templates for common products."""
        st.markdown("### ⚡ Швидке додавання")
        
        templates = {
            "Хліб білий": (50, 12.0),
            "Яблуко": (150, 14.0),
            "Банан": (120, 20.0),
            "Рис": (100, 28.0),
            "Картопля": (200, 17.0)
        }
        
        cols = st.columns(len(templates))
        for i, (name, (weight, carbs)) in enumerate(templates.items()):
            with cols[i]:
                if st.button(f"{name}\n{weight}г", use_container_width=True):
                    self.add_product(name, carbs, weight)
    
    def render_meal_table(self) -> None:
        """Render meal data table with enhanced features."""
        if not st.session_state.meal_data:
            st.info("👆 Додайте продукти, щоб побачити розрахунок")
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
    
    def render_results(self) -> None:
        """Render calculation results with enhanced visualization."""
        if not st.session_state.meal_data:
            return
        
        totals = self.calculate_totals()
        
        # Main result card
        st.markdown(f"""
        <div class="result-card">
            <h2 style="margin:0; font-size: 2.5rem;">{totals['total_bu']} ХО</h2>
            <p style="margin:0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">{totals['total_carbs']} г вуглеводів</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Середнє ХО на продукт", 
                f"{totals['total_bu'] / len(st.session_state.meal_data):.2f}",
                delta=None
            )
        
        with col2:
            insulin_ratio = st.number_input(
                "Співвідношення інсуліну",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                value=1.0,
                help="Одиниці інсуліну на 1 ХО"
            )
        
        with col3:
            insulin_needed = totals['total_bu'] * insulin_ratio
            st.metric(
                "Потрібно інсуліну", 
                f"{insulin_needed:.1f} од.",
                delta=None
            )
    
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
        """Main application entry point."""
        # Apply custom CSS
        st.markdown(self.get_custom_css(), unsafe_allow_html=True)
        
        # Render components
        self.render_sidebar()
        self.render_main_interface()
        self.render_footer()


# Initialize and run the application
if __name__ == "__main__":
    app = DiabetesCalculator()
    app.run()