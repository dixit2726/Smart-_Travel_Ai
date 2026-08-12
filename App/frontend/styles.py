import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* Global Theme Settings */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F8FAFC;
        }

        .main {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
            background-attachment: fixed;
        }

        /* Glassmorphism Card System */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-3px);
            box-shadow: 0 16px 40px rgba(56, 189, 248, 0.15);
        }

        .glass-card-interactive {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 18px;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .glass-card-interactive:hover {
            background: rgba(51, 65, 85, 0.8);
            border-color: #38BDF8;
            transform: scale(1.02);
        }

        /* Hero Banner styling */
        .hero-banner {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.80) 100%),
                        url('https://images.unsplash.com/photo-1590050752117-238cb0fb12b1?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            border-radius: 24px;
            padding: 60px 45px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            margin-bottom: 36px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }

        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            line-height: 1.15;
            background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 60%, #818CF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            color: #CBD5E1;
            font-weight: 400;
            line-height: 1.6;
            max-width: 720px;
            margin-bottom: 28px;
        }

        .home-section-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #F8FAFC;
            margin-top: 40px;
            margin-bottom: 8px;
            letter-spacing: -0.01em;
        }

        .home-section-subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            margin-bottom: 24px;
        }

        /* 6 Smart Feature Cards */
        .feature-card {
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 22px 24px;
            height: 195px !important;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            transition: all 0.3s ease;
            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        .feature-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-4px);
            box-shadow: 0 16px 36px rgba(14, 165, 233, 0.15);
            background: rgba(30, 41, 59, 0.85);
        }

        .feature-icon {
            font-size: 2.0rem;
            margin-bottom: 10px;
            display: inline-block;
        }

        .feature-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 6px;
        }

        .feature-desc {
            font-size: 0.88rem;
            color: #94A3B8;
            line-height: 1.45;
            margin: 0;
        }

        /* 5-Step Flow Cards */
        .step-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 20px 16px;
            text-align: center;
            height: 100%;
            position: relative;
            transition: all 0.25s ease;
        }

        .step-card:hover {
            border-color: #38BDF8;
            background: rgba(30, 41, 59, 0.9);
            transform: translateY(-2px);
        }

        .step-number {
            display: inline-block;
            background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
            color: #FFFFFF;
            font-size: 0.8rem;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 12px;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }

        .step-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 4px;
        }

        /* 4 Experience Category Image Cards */
        .exp-card {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
            transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
            height: 280px;
            margin-bottom: 15px;
        }

        .exp-card:hover {
            transform: translateY(-5px) scale(1.01);
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 20px 45px rgba(14, 165, 233, 0.25);
        }

        .exp-card-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }

        .exp-card:hover .exp-card-img {
            transform: scale(1.06);
        }

        .exp-card-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            top: 0;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.1) 0%, rgba(15, 23, 42, 0.92) 80%);
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }

        .exp-card-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 6px;
        }

        .exp-card-desc {
            font-size: 0.88rem;
            color: #CBD5E1;
            line-height: 1.4;
            margin: 0;
        }

        /* Map Concept Preview Container */
        .map-preview-box {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 20px;
            padding: 30px;
            margin-top: 15px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }

        /* AI Pipeline Visualization Box */
        .ai-pipeline-container {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 20px;
            padding: 28px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        /* Final CTA Banner */
        .cta-banner {
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.2) 0%, rgba(37, 99, 235, 0.25) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 24px;
            padding: 45px 36px;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 45px rgba(0,0,0,0.4);
        }

        /* Stat Metrics Cards */
        .stat-card {
            background: rgba(15, 23, 42, 0.7);
            border-left: 4px solid #38BDF8;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        .stat-card-green { border-left-color: #10B981; }
        .stat-card-amber { border-left-color: #F59E0B; }
        .stat-card-purple { border-left-color: #8B5CF6; }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
        }

        .stat-label {
            font-size: 0.875rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Custom Badges */
        .badge-green {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        .badge-amber {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        .badge-red {
            background: rgba(239, 68, 68, 0.15);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        .badge-blue {
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        /* Commercial Button Styling */
        div.stButton > button {
            background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 14px rgba(14, 165, 233, 0.3) !important;
            transition: all 0.25s ease !important;
            width: 100%;
        }

        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5) !important;
            background: linear-gradient(135deg, #38BDF8 0%, #3B82F6 100%) !important;
        }

        /* Secondary Button Override */
        div.stDownloadButton > button {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #F8FAFC !important;
            border-radius: 12px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
        }

        div.stDownloadButton > button:hover {
            background: rgba(51, 65, 85, 1) !important;
            border-color: #38BDF8 !important;
        }

        /* Step Wizard Indicator Bar */
        .step-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 14px 20px;
            margin-bottom: 25px;
        }

        .step-item {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #64748B;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .step-item.active {
            color: #38BDF8;
        }

        .step-item.completed {
            color: #34D399;
        }

        .step-number {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255, 255, 255, 0.1);
            font-size: 0.85rem;
        }

        .step-item.active .step-number {
            background: #0EA5E9;
            color: #FFFFFF;
            box-shadow: 0 0 12px rgba(14, 165, 233, 0.6);
        }

        .step-item.completed .step-number {
            background: #10B981;
            color: #FFFFFF;
        }

        /* Selected Spot Pill Badges */
        .spot-pill {
            background: rgba(14, 165, 233, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38BDF8;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin: 4px;
        }

        /* Tab Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 8px;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 10px 20px !important;
            color: #94A3B8 !important;
            font-weight: 600 !important;
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(30, 41, 59, 1) !important;
            color: #38BDF8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
        }

        /* Streamlit Option Menu Container Styling */
        .nav-link {
            font-weight: 600 !important;
            border-radius: 10px !important;
        }

        /* Problem Card System */
        .problem-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(239, 68, 68, 0.25);
            border-radius: 16px;
            padding: 22px;
            height: 100%;
            transition: all 0.3s ease;
        }

        .problem-card:hover {
            border-color: rgba(239, 68, 68, 0.5);
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(239, 68, 68, 0.15);
            background: rgba(30, 41, 59, 0.85);
        }

        .problem-icon {
            font-size: 2rem;
            margin-bottom: 10px;
            display: inline-block;
        }

        .problem-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 6px;
        }

        .problem-desc {
            font-size: 0.9rem;
            color: #94A3B8;
            line-height: 1.5;
            margin: 0;
        }

        /* Why Smart Tourism Card */
        .why-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.88) 100%),
                        url('https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            border-radius: 20px;
            padding: 36px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 25px;
        }

        /* Pipeline Node styling */
        .pipeline-node {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 14px 18px;
            border-radius: 14px;
            font-weight: 700;
            font-size: 0.92rem;
            color: #F8FAFC;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            transition: all 0.25s ease;
        }

        .pipeline-node:hover {
            border-color: #38BDF8;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(56, 189, 248, 0.25);
        }

        .pipeline-arrow {
            color: #38BDF8;
            font-size: 1.3rem;
            font-weight: 800;
        }

        /* Hide Streamlit Header / Footer branding while keeping sidebar controls fully functional */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Make top header background transparent so background gradient is continuous */
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Ensure sidebar toggle icons have high visibility and clear cursor pointer */
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button,
        button[aria-label="Collapse sidebar"],
        button[aria-label="Expand sidebar"] {
            color: #38BDF8 !important;
            cursor: pointer !important;
        }

        /* Sidebar Logo Container Styling for High Contrast */
        section[data-testid="stSidebar"] div[data-testid="stImage"] img,
        [data-testid="stSidebar"] [data-testid="stImage"] img,
        [data-testid="stSidebar"] img {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
            padding: 12px 18px !important;
            border-radius: 16px !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
            margin: 0 auto 12px auto !important;
            display: block !important;
            object-fit: contain !important;
        }

        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: rgba(56, 189, 248, 0.15) !important;
        }
        </style>
    """, unsafe_allow_html=True)
