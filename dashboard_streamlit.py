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


    # 📈 АНАЛИЗ СЕЗОННОСТИ И ТРЕНДОВ
    st.header("📈 Анализ сезонности и трендов")
    
    # Создаем дополнительные колонки для анализа
    df['created_date_only'] = df['created_date'].dt.date
    df['created_week'] = df['created_date'].dt.isocalendar().week
    df['created_month'] = df['created_date'].dt.month
    df['created_day_of_week'] = df['created_date'].dt.day_name()
    df['created_hour'] = df['created_date'].dt.hour
    
    # 1. Еженедельный объем тикетов (последние 6 недель)
    st.subheader("Еженедельный объем созданных тикетов")
    
    weekly_volume = df.resample('W', on='created_date')['ticket_id'].count().tail(6)
    weekly_labels = weekly_volume.index.strftime('%Y-%m-%d')
    
    fig_weekly, ax_weekly = plt.subplots(figsize=(10, 6))
    ax_weekly = sns.lineplot(x=weekly_labels, y=weekly_volume.values, 
                           marker='o', markersize=8, linewidth=2.5,
                           color='steelblue', ax=ax_weekly)
    
    ax_weekly.set_title('Еженедельный объем созданных тикетов (последние 6 недель)\n', fontsize=16)
    ax_weekly.set_xlabel('\nНеделя')
    ax_weekly.set_ylabel('Количество тикетов\n')
    ax_weekly.grid(True, alpha=0.3)
    
    # Подписываем значения точек
    for i, (x, y) in enumerate(zip(weekly_labels, weekly_volume.values)):
        ax_weekly.annotate(f'{int(y)}', 
                          (x, y),
                          xytext=(0, 10), 
                          textcoords='offset points',
                          ha='center', va='bottom', 
                          fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_weekly)
    
    # 2. Ежемесячный объем тикетов
    st.subheader("Ежемесячный объем созданных тикетов")
    
    monthly_volume = df.resample('M', on='created_date')['ticket_id'].count()
    monthly_labels = monthly_volume.index.strftime('%Y-%m')
    
    fig_monthly, ax_monthly = plt.subplots(figsize=(10, 6))
    ax_monthly = sns.lineplot(x=monthly_labels, y=monthly_volume.values, 
                            marker='s', markersize=8, linewidth=2.5,
                            color='darkorange', ax=ax_monthly)
    
    ax_monthly.set_title('Ежемесячный объем созданных тикетов\n', fontsize=16)
    ax_monthly.set_xlabel('\nМесяц')
    ax_monthly.set_ylabel('Количество тикетов\n')
    ax_monthly.grid(True, alpha=0.3)
    
    # Подписываем значения точек
    for i, (x, y) in enumerate(zip(monthly_labels, monthly_volume.values)):
        ax_monthly.annotate(f'{int(y)}', 
                           (x, y),
                           xytext=(0, 10), 
                           textcoords='offset points',
                           ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_monthly)
    
    # 3. Распределение тикетов по времени суток
    st.subheader("Распределение тикетов по времени суток")
    
    time_intervals = {
        'Ночь (0-6)': df[df['created_hour'].between(0, 6)]['ticket_id'].count(),
        'Утро (7-10)': df[df['created_hour'].between(7, 10)]['ticket_id'].count(),
        'День (11-18)': df[df['created_hour'].between(11, 17)]['ticket_id'].count(),
        'Вечер (19-23)': df[df['created_hour'].between(19, 23)]['ticket_id'].count()
    }
    
    fig_time, ax_time = plt.subplots(figsize=(8, 6))
    colors = ['lightpink', 'lightblue', 'lightgreen', 'lightyellow']
    
    wedges, texts, autotexts = ax_time.pie(
        time_intervals.values(), 
        labels=time_intervals.keys(), 
        autopct='%1.1f%%', 
        colors=colors,
        startangle=90
    )
    
    ax_time.set_title('Распределение тикетов по времени суток\n', fontsize=16)
    
    # Увеличиваем шрифт для подписей
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    st.pyplot(fig_time)

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

    # 📊 АНАЛИЗ ВРЕМЕНИ В СТАТУСАХ
    st.header("📊 Анализ времени в статусах")
    
    # Определяем статусы
    closed_statuses = ['Выполнено', 'Отменен']
    active_statuses = ['Ожидает обработки', 'В работе', 'В разработке', 'На согласовании', 'Ожидается ответ пользователя']
    
    # Берем только тикеты в финальных статусах
    closed_tickets = df_status[df_status['status'].isin(closed_statuses)]['ticket_id'].unique()
    df_closed = df_status[df_status['ticket_id'].isin(closed_tickets)].copy()
    
    # Сортируем и рассчитываем время между сменами статусов
    df_closed = df_closed.sort_values(['ticket_id', 'changed_date'])
    df_closed['next_changed_date'] = df_closed.groupby('ticket_id')['changed_date'].shift(-1)
    
    # Для последней записи каждого тикета используем максимальную дату
    max_dates = df_closed.groupby('ticket_id')['changed_date'].max()
    for ticket_id in closed_tickets:
        last_idx = df_closed[df_closed['ticket_id'] == ticket_id].index[-1]
        df_closed.loc[last_idx, 'next_changed_date'] = max_dates[ticket_id]
    
    # Фильтруем только активные статусы и рассчитываем время
    df_active_times = df_closed[df_closed['status'].isin(active_statuses)].copy()
    df_active_times['time_in_status'] = (df_active_times['next_changed_date'] - df_active_times['changed_date']).dt.total_seconds() / 3600
    
    # Суммируем время по активным статусам
    time_per_active_status = df_active_times.groupby('status')['time_in_status'].sum()
    total_active_time = time_per_active_status.sum()
    percentage_time_per_status = (time_per_active_status / total_active_time * 100).round(2)
    
    # пайчарт
    st.subheader("Распределение времени по активным статусам")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Pastel1(range(len(percentage_time_per_status)))
    
    wedges, texts, autotexts = ax.pie(percentage_time_per_status.values, 
                                      labels=percentage_time_per_status.index,
                                      autopct='%1.1f%%', 
                                      colors=colors,
                                      startangle=90)
    
    ax.set_title('Распределение времени по активным статусам закрытых тикетов', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)

    # 👤📈 KPI ИСПОЛНИТЕЛЕЙ
    st.header("👤📈 KPI исполнителей")
    
    # Процент тикетов, решенных тем же исполнителем
    st.subheader("Процент тикетов, решенных тем же исполнителем")
    
    # Фильтруем данные где first_assignee тот же, что и last_assignee
    same_assignee_tickets = df[df['first_assignee'] == df['last_assignee']]
    ticket_count_per_assignee = same_assignee_tickets['first_assignee'].value_counts()
    
    # Общее количество тикетов по каждому исполнителю
    total_tickets_per_assignee = df['first_assignee'].value_counts()
    
    # Вычисляем процентное соотношение
    percentage_solved_same = (ticket_count_per_assignee / total_tickets_per_assignee) * 100
    percentage_solved_same = percentage_solved_same.sort_values(ascending=False)
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # градиент серого
    n_bars = len(percentage_solved_same)
    gray_colors = [f'#{i:02x}{i:02x}{i:02x}' for i in np.linspace(180, 100, n_bars).astype(int)]
    
    bars = ax.bar(percentage_solved_same.index, percentage_solved_same.values, 
                 color=gray_colors, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Настройки графика
    ax.set_title('Процент тикетов, решенных тем же исполнителем\n', fontsize=16, pad=20)
    ax.set_xlabel('Исполнитель', fontsize=12)
    ax.set_ylabel('% решенных самостоятельно\n', fontsize=12)
    ax.set_xticklabels(percentage_solved_same.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, percentage_solved_same.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Медианное время решения по исполнителям
    st.subheader("Медианное время решения тикетов по исполнителям")
    
    # Рассчитываем медианное время решения для тикетов, где first_assignee = last_assignee
    same_assignee_tickets = df[df['first_assignee'] == df['last_assignee']]
    median_resolution_time = same_assignee_tickets.groupby('first_assignee')['resolution_in_hours'].median()
    median_resolution_time = median_resolution_time.sort_values(ascending=True)  # Сортируем для лучшего отображения
    
    # отдельный DataFrame для удобства
    median_resolution_df = pd.DataFrame({
        'Исполнитель': median_resolution_time.index,
        'median_time': median_resolution_time.values
    }).sort_values('median_time', ascending=True)
    
    # барплот
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # градиент серого
    n_bars = len(median_resolution_df)
    gray_colors = [f'#{i:02x}{i:02x}{i:02x}' for i in np.linspace(180, 100, n_bars).astype(int)]
    
    bars = ax.barh(median_resolution_df['Исполнитель'], median_resolution_df['median_time'],
                  color=gray_colors, edgecolor='black', linewidth=1, alpha=0.8)
    
    # Настройки
    ax.set_title('Медианное время решения тикетов по исполнителям\n(first_assignee = last_assignee)', 
                fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Медианное время решения (часы)', fontsize=12)
    ax.set_ylabel('Исполнитель', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, median_resolution_df['median_time']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{value:.1f}ч', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Эффективность по меткам исполнителя
    st.subheader("Распределение меток по исполнителям")
    
    same_assignee_tickets = df[df['first_assignee'] == df['last_assignee']]

    # Топ меток по отфильтрованным данным
    top_n_tags = same_assignee_tickets['tag'].value_counts().head(15).index.tolist()

    # Группируем отфильтрованные данные по исполнителям и меткам
    grouped_data = same_assignee_tickets.groupby(['first_assignee', 'tag']).size().unstack(fill_value=0)

    # Оставляем только топ метки, остальные определим в "Other"
    available_top_tags = [tag for tag in top_n_tags if tag in grouped_data.columns]
    grouped_data_top = grouped_data[available_top_tags].copy()

    # Добавляем колонку Other для остальных меток
    other_tags = [col for col in grouped_data.columns if col not in available_top_tags]
    if other_tags:
        grouped_data_top['Other'] = grouped_data[other_tags].sum(axis=1)
    else:
        grouped_data_top['Other'] = 0

    # Сортируем исполнителей по общему количеству тикетов
    grouped_data_top = grouped_data_top.loc[grouped_data_top.sum(axis=1).sort_values(ascending=False).index]

    # график
    fig, ax = plt.subplots(figsize=(14, 10))

    # Цветовая карта
    colors = plt.cm.Set3(np.linspace(0, 1, len(grouped_data_top.columns)))
    cmap = ListedColormap(colors)

    # барплот с накоплением
    bars = grouped_data_top.plot(kind='barh', stacked=True, ax=ax, colormap=cmap, alpha=0.8)

    ax.set_title('Распределение метки по исполнителям\n(first_assignee = last_assignee)', 
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Количество тикетов', fontsize=12)
    ax.set_ylabel('Исполнитель', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Легенду выносим справа
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
              title='Метки (tags)', title_fontsize=11, fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Место для легенды
    st.pyplot(fig)

#вход в программу        
if __name__ == "__main__":
    main()
