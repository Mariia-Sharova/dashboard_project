#streamlit==1.49.0
#pandas==1.5.3
#numpy==1.24.3
#plotly==5.9.0
#matplotlib==3.7.1
#seaborn==0.12.2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# стиль
plt.style.use('default')
sns.set(style="whitegrid", palette="gray")

# кешируем для увеличения производительности
@st.cache_data
def load_data():
    df = pd.read_csv('df.csv', parse_dates=['created_date', 'resolved_date'])
    df = df.drop(df.columns[0], axis=1)
    df['last_status'] = df['last_status'].replace('В работу', 'В работе')
    
    # время решения в часах
    df['resolution_in_hours'] = (df['resolved_date'] - df['created_date']).dt.total_seconds() / 3600
    
    # проверка соблюдения SLA
    df['sla_violation'] = df['resolution_in_hours'] > df['sla_hours']
    df['sla_compliance'] = ~df['sla_violation']
    
    return df

# Загрузка данных второго датафрейма
@st.cache_data
def load_status_data():
    return pd.read_csv('ticket_status_history_detailed_last_ver.csv', parse_dates=['changed_date'])

# Функция анализа данных статусов
def analyze_status_data(df_status):
    df_sorted = df_status.sort_values(by=['ticket_id', 'changed_date'])
    
    # Статусы
    closed_statuses = ['Выполнено', 'Отменен']
    active_statuses = ['В работу', 'В работе', 'В разработке', 'На согласовании', 'Ожидается ответ пользователя']
    
    # отбираем закрытые и активные статусы
    df_sorted['is_closed'] = df_sorted['status'].isin(closed_statuses)
    df_sorted['is_active'] = df_sorted['status'].isin(active_statuses)
    
    # находим переоткрытые тикеты
    df_sorted['next_is_active'] = df_sorted.groupby('ticket_id')['is_active'].shift(-1)
    reopened_mask = df_sorted['is_closed'] & df_sorted['next_is_active']
    reopened_tickets = df_sorted.loc[reopened_mask, 'ticket_id'].unique().tolist()
    
    # Анализ по месяцам и закрытые тикеты
    df_sorted['month'] = df_sorted['changed_date'].dt.to_period('M')
    closed_tickets = df_sorted[df_sorted['status'].isin(closed_statuses)]
    closed_per_month = closed_tickets.groupby('month')['ticket_id'].nunique().rename('closed_tickets')
    
    # переоткрытые тикеты по месяцам
    reopened_df = df_sorted[df_sorted['ticket_id'].isin(reopened_tickets)]
    reopened_per_month = reopened_df.groupby('month')['ticket_id'].nunique().rename('reopened_tickets')
    
    # Собираем итоговую таблицу
    result_df = pd.concat([closed_per_month, reopened_per_month], axis=1).fillna(0)
    result_df['percentage_reopened'] = (result_df['reopened_tickets'] / result_df['closed_tickets'] * 100).fillna(0)
    
    # Ответственные за повторные открытия исполнители
    assignee_reopened = df_sorted[
        (df_sorted['ticket_id'].isin(reopened_tickets)) & 
        (df_sorted['status'].isin(closed_statuses))
    ].groupby('ticket_id').last().reset_index()[['ticket_id', 'assignee']]
    
    assignee_counts = assignee_reopened['assignee'].value_counts()
    
    return {
        'result_df': result_df,
        'assignee_counts': assignee_counts,
        'reopened_tickets': reopened_tickets
    }

# Основное приложение
def main():
    # Загрузка данных
    df = load_data()
    df_status = load_status_data()
    status_analysis = analyze_status_data(df_status)
    
    # Сайдбар, настраиваем фильтры
    st.sidebar.title("Фильтры")

    # Фильтр по датам
    min_date = df['created_date'].min()
    max_date = df['created_date'].max()

    date_range = st.sidebar.date_input(
        "Диапазон дат создания",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['created_date'] >= pd.Timestamp(start_date)) & #оставляет даты, которые НЕ раньше начала диапазона
                (df['created_date'] <= pd.Timestamp(end_date))]  #ставляет даты, которые НЕ позже конца диапазона

    # Фильтр по приоритетам
    priorities = st.sidebar.multiselect(
        "Приоритет",
        options=sorted(df['priority'].unique()),
        default=sorted(df['priority'].unique())
    )

    if priorities:
        df = df[df['priority'].isin(priorities)]

    # Фильтр по меткам
    tags = st.sidebar.multiselect(
        "Метки",
        options=sorted(df['tag'].dropna().unique()),
        default=sorted(df['tag'].dropna().unique())
    )

    if tags:
        df = df[df['tag'].isin(tags)]

    # Фильтр по исполнителям
    assignees = st.sidebar.multiselect(
        "Исполнители",
        options=sorted(df['last_assignee'].unique()),
        default=sorted(df['last_assignee'].unique())
    )

    if assignees:
        df = df[df['last_assignee'].isin(assignees)]
    
    # Кнопка обновления датафреймов
    st.sidebar.header("Дополнительные данные")
    if st.sidebar.button("🔄 Обновить историю статусов", key="refresh_status"):
        st.cache_data.clear()
        st.rerun()

    # Заголовок дашборда
    st.title("Анализ тикетов поддержки")

    # KPI баны
    col1, col2 = st.columns(2) #создали два столбца для Банов, будет среднее и медианное время решения тикетов

    with col1:
        avg_resolution = df['resolution_in_hours'].mean()
        st.metric("Среднее время решения (часы)", f"{avg_resolution:.1f}")

    with col2:
        median_resolution = df['resolution_in_hours'].median()
        st.metric("Медианное время решения (часы)", f"{median_resolution:.1f}")

    # 📊 Основные KPI, эффективность процессов:
    st.header("📊 Основные KPI, эффективность процессов:")

    # Процент соблюдения SLA, процент просрочек
    st.subheader("Процент соблюдения SLA, процент просрочек")

    violated = (df['sla_compliance'] == False).sum()
    complied = (df['sla_compliance'] == True).sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts = [violated, complied]
    labels = ['Нарушено', 'Соблюдено']
    colors = ['red', 'green']

    ax.bar(labels, counts, color=colors, alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.set_title('Проверка на соблюдение SLA')
    ax.set_xlabel('Соблюдено SLA')
    ax.set_ylabel('Количество тикетов')

    #подписи значений над столбцами
    for i, count in enumerate(counts):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')

    ax.grid(False)
    st.pyplot(fig)

    # Доля соблюдения SLA по приоритетам
    sla_priority = df.groupby('priority')['sla_compliance'].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.light_palette("gray", n_colors=len(sla_priority))
    ax = sla_priority.plot(kind='bar', color=colors, edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_title('Доля соблюдения SLA по приоритетам')
    ax.set_xlabel('Приоритет')
    ax.set_ylabel('Доля соблюдения SLA (%)')
    ax.set_xticklabels(sla_priority.index, rotation=0)

    for i, v in enumerate(sla_priority):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)#подписи над столбцом

    ax.grid(False)
    st.pyplot(fig)

    # 📅 Метрики загрузки
    st.header("📅 Метрики загрузки")

    # Общее количество тикетов
    st.subheader("Общее количество тикетов")
    st.write(f"Всего тикетов: {df['ticket_id'].count()}")

    # Распределение тикетов по приоритетам
    st.subheader("Распределение тикетов по приоритетам")

    ticket_counts = df['priority'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ticket_counts.index, ticket_counts.values, alpha=0.7, edgecolor='black')
    ax.set_title('Распределение тикетов по приоритетам')
    ax.set_xlabel('Приоритет')
    ax.set_ylabel('Количество тикетов')

    for i, count in enumerate(ticket_counts.values):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')#подписи над столбцом

    ax.grid(False)
    st.pyplot(fig)

    # Распределение тикетов по меткам
    st.subheader("Распределение тикетов по меткам")

    tag_counts = df['tag'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(tag_counts)), tag_counts.values, alpha=0.7, edgecolor='black')
    ax.set_title('Распределение тикетов по меткам')
    ax.set_xlabel('Метки')
    ax.set_ylabel('Количество тикетов')
    ax.set_xticks(range(len(tag_counts)))
    ax.set_xticklabels(tag_counts.index, rotation=45, ha='right')

    for i, count in enumerate(tag_counts.values):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold', fontsize=8)

    ax.grid(False)
    st.pyplot(fig)


    # 🔄 АНАЛИЗ ПОВТОРНЫХ ОБРАЩЕНИЙ
    st.header("🔄 Анализ переоткрытых тикетов")
    
    if status_analysis is not None:
        # 1. График процента повторно открытых тикетов
        st.subheader("Процент повторно открытых тикетов по месяцам")
        
        monthly_data = status_analysis['result_df'].reset_index()
        monthly_data['month_str'] = monthly_data['month'].astype(str)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        n_bars = len(monthly_data)
        gray_colors = [f'#{i:02x}{i:02x}{i:02x}' for i in np.linspace(180, 100, n_bars).astype(int)]
        
        ax1 = sns.barplot(x='month_str', y='percentage_reopened', data=monthly_data,
                         palette=gray_colors, 
                         edgecolor='black',
                         linewidth=0.5,
                         ax=ax1)
        
        ax1.set_title('Процент повторно открытых тикетов по месяцам\n', fontsize=16)
        ax1.set_xlabel('\nМесяц')
        ax1.set_ylabel('Процент повторного открытия\n')
        ax1.grid(False)
        
        for p in ax1.patches:
            ax1.annotate(f'{p.get_height():.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # 2. График распределения ответственных
        st.subheader("Топ ответственных за повторно открытые тикеты")
        
        if len(status_analysis['assignee_counts']) > 0:
            top_assignees = status_analysis['assignee_counts'].head(10)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            n_bars = len(top_assignees)
            gray_colors = [f'#{i:02x}{i:02x}{i:02x}' for i in np.linspace(180, 100, n_bars).astype(int)]
            
            ax2 = sns.barplot(x=top_assignees.index, y=top_assignees.values,
                             palette=gray_colors, 
                             edgecolor='black',
                             linewidth=0.5,
                             ax=ax2)
            
            ax2.set_title('Топ-10 ответственных за повторно открытые тикеты\n', fontsize=16)
            ax2.set_xlabel('\nОтветственный')
            ax2.set_ylabel('Количество тикетов\n')
            ax2.grid(False)
            
            for p in ax2.patches:
                ax2.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Данные по ответственным не найдены")
        
        # 3. График абсолютных значений
        st.subheader("Динамика закрытых и повторно открытых тикетов")
        
        stacked_data = status_analysis['result_df'][['closed_tickets', 'reopened_tickets']].reset_index()
        stacked_data['month_str'] = stacked_data['month'].astype(str)
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        width = 0.6
        
        light_gray = '#d3d3d3'
        dark_gray = '#808080'
        
        bars1 = ax3.bar(stacked_data['month_str'], stacked_data['closed_tickets'], width,
                       color=light_gray, 
                       edgecolor='black',
                       linewidth=0.5, 
                       label='Всего закрыто')
        
        bars2 = ax3.bar(stacked_data['month_str'], stacked_data['reopened_tickets'], width,
                       color=dark_gray, 
                       edgecolor='black',
                       linewidth=0.5, 
                       label='Повторно открыто',
                       bottom=stacked_data['closed_tickets'] - stacked_data['reopened_tickets'])
        
        ax3.set_title('Динамика закрытых и повторно открытых тикетов по месяцам\n', fontsize=16)
        ax3.set_xlabel('\nМесяц')
        ax3.set_ylabel('Количество тикетов\n')
        ax3.legend()
        ax3.grid(False)
        
        for i, (bar, value) in enumerate(zip(bars1, stacked_data['closed_tickets'])):
            ax3.text(bar.get_x() + bar.get_width()/2, value + 1, f'{value}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for i, (bar, value) in enumerate(zip(bars2, stacked_data['reopened_tickets'])):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, 
                        stacked_data['closed_tickets'].iloc[i] - value/2, 
                        f'{value}', ha='center', va='center', fontsize=9, 
                        fontweight='bold', color='white')
        
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # 4. Тренд повторного открытия
        st.subheader("Тренд процента переоткрытия тикетов")
        
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        plt.plot(monthly_data['month_str'], monthly_data['percentage_reopened'], 
                 marker='o', 
                 linewidth=2, 
                 markersize=6,
                 color='#666666',
                 markerfacecolor='white',
                 markeredgecolor='black',
                 markeredgewidth=0.5)
        
        plt.title('Тренд процента переоткрытия тикетов\n', fontsize=16)
        plt.xlabel('\nМесяц')
        plt.ylabel('Процент повторного открытия\n')
        plt.grid(True, alpha=0.3)
        
        for i, (month, value) in enumerate(zip(monthly_data['month_str'], monthly_data['percentage_reopened'])):
            plt.annotate(f'{value:.1f}%', 
                        (month, value),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', 
                        va='bottom',
                        fontsize=9,
                        fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Добавляем инфо в сайдбар, под кнопку обновлений
        tickets_with_tags = df['tag'].notna().sum()
        tickets_without_tags = df['tag'].isna().sum()
        
        st.sidebar.info(f"Нарушено SLA: {violated}")
        st.sidebar.info(f"Соблюдено SLA: {complied}")
        #st.sidebar.info(f"Всего тикетов в истории: {len(df_status['ticket_id'].unique())}")
        st.sidebar.info(f"Повторно открытых тикетов: {len(status_analysis['reopened_tickets'])}")
        st.sidebar.info(f"Тикетов с метками: {tickets_with_tags}")
        st.sidebar.info(f"Тикетов без меток: {tickets_without_tags}")
    else:
        st.warning("Данные для анализа повторных обращений недоступны")

#вход в программу        
if __name__ == "__main__":
    main()

