"""
Streamlit App for Intelligent Traffic System.

Simple web interface to interact with the API endpoints.
"""

import streamlit as st
import requests
from PIL import Image
import io
import json
from typing import Optional, Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Intelligent Traffic System",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚦 Intelligent Traffic System</h1>', unsafe_allow_html=True)

# Sidebar - API Status Check
with st.sidebar:
    st.header("🔧 API Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
            health_data = response.json()
            st.info(f"Status: {health_data.get('status', 'unknown')}")
        else:
            st.error("❌ API Error")
    except requests.exceptions.RequestException:
        st.error("❌ API Not Running")
        st.warning("Please start the API server:\n```bash\npython run_api.py\n```")
        st.stop()
    
    st.divider()
    st.markdown("### 📚 API Documentation")
    st.markdown(f"[Swagger UI]({API_BASE_URL}/docs)")
    st.markdown(f"[ReDoc]({API_BASE_URL}/redoc)")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🛣️ Road Detection",
    "🚦 Traffic Sign Detection",
    "📊 Traffic Flow Prediction",
    "🔧 Road Condition Detection",
    "🔄 Complete Analysis"
])

# Helper function to make API requests
def call_api(
    endpoint: str,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Make API request and handle response.
    
    Args:
        endpoint: API endpoint path.
        files: Files to upload.
        data: Form data.
        
    Returns:
        Response data or None if error.
    """
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            # Check if response is an image
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                # Get result data from headers
                result_data = response.headers.get('X-Result-Data')
                if result_data:
                    return {
                        'image': response.content,
                        'data': json.loads(result_data)
                    }
                return {'image': response.content}
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"❌ API Error: {error_detail}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection Error: {str(e)}")
        return None


# Tab 1: Road Detection
with tab1:
    st.header("🛣️ Road & Lane Detection")
    st.markdown("Detect and track road lanes in real-time")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Road Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image with visible lane markings",
            key="road_upload"
        )
        
        return_image = st.checkbox("Show processed image", value=True, key="road_return_image")
        
        if st.button("🔍 Detect Lanes", type="primary", key="road_detect_btn"):
            if uploaded_file is not None:
                with st.spinner("Processing image..."):
                    files = {'file': uploaded_file.getvalue()}
                    data = {'return_image': str(return_image).lower()}
                    
                    result = call_api("/api/v1/road/detect", files=files, data=data)
                    
                    if result:
                        if 'image' in result:
                            # Display processed image
                            st.image(result['image'], caption="Processed Image with Lane Detection", use_container_width=True)
                            result_data = result.get('data', {})
                        else:
                            result_data = result
                        
                        # Display results
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Detection Complete!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Lane Detected", "✅ Yes" if result_data.get('lane_detected') else "❌ No")
                        with col_b:
                            st.metric("Confidence", f"{result_data.get('lane_confidence', 0):.2%}")
                        with col_c:
                            st.metric("Departure Warning", "⚠️ Yes" if result_data.get('departure_warning') else "✅ No")
                        
                        st.json(result_data)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please upload an image first")
        
        # Example images
        st.markdown("### 📸 Test Images")
        st.info("Try with: `data/test_images/lane.jpeg`, `lane2.jpeg`, `lane3.jpeg`, `lane4.jpeg`")


# Tab 2: Traffic Sign Detection
with tab2:
    st.header("🚦 Traffic Sign Detection")
    st.markdown("Recognize and classify traffic signs using AI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Traffic Sign Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing traffic signs",
            key="sign_upload"
        )
        
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Filter detections by confidence score",
            key="sign_confidence"
        )
        
        return_image = st.checkbox("Show processed image", value=True, key="sign_return_image")
        
        if st.button("🔍 Detect Signs", type="primary", key="sign_detect_btn"):
            if uploaded_file is not None:
                with st.spinner("Detecting traffic signs..."):
                    files = {'file': uploaded_file.getvalue()}
                    data = {
                        'min_confidence': str(min_confidence),
                        'return_image': str(return_image).lower()
                    }
                    
                    result = call_api("/api/v1/traffic-sign/detect", files=files, data=data)
                    
                    if result:
                        if 'image' in result:
                            st.image(result['image'], caption="Processed Image with Sign Detection", use_container_width=True)
                            result_data = result.get('data', {})
                        else:
                            result_data = result
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Found {result_data.get('signs_detected', 0)} traffic signs")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Signs Detected", result_data.get('signs_detected', 0))
                            st.metric("Total Found", result_data.get('total_signs_found', 0))
                        with col_b:
                            st.metric("Processing Time", f"{result_data.get('processing_time', 0):.3f}s")
                            st.metric("Min Confidence", result_data.get('min_confidence_used', 0))
                        
                        # Display detected signs
                        if result_data.get('signs'):
                            st.subheader("📋 Detected Signs")
                            for i, sign in enumerate(result_data['signs'], 1):
                                with st.expander(f"Sign {i}: {sign.get('class_name', 'Unknown')} ({sign.get('confidence', 0):.2%})"):
                                    st.json(sign)
                        
                        st.json(result_data)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please upload an image first")
        
        st.markdown("### 📸 Test Images")
        st.info("Try with: `data/test_images/0.png`, `all-signs.png`, `Sample_2.png`, `Sample_3.png`")


# Tab 3: Traffic Flow Prediction
with tab3:
    st.header("📊 Traffic Flow Prediction")
    st.markdown("Predict traffic congestion based on vehicle counts")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Vehicle Counts")
        car_count = st.number_input("🚗 Cars", min_value=0, value=15, step=1, key="flow_cars")
        bike_count = st.number_input("🏍️ Bikes", min_value=0, value=3, step=1, key="flow_bikes")
        bus_count = st.number_input("🚌 Buses", min_value=0, value=2, step=1, key="flow_buses")
        truck_count = st.number_input("🚛 Trucks", min_value=0, value=5, step=1, key="flow_trucks")
        
        total_count = car_count + bike_count + bus_count + truck_count
        st.metric("📈 Total Vehicles", total_count)
        
        if st.button("🔮 Predict Traffic", type="primary", key="flow_predict_btn"):
            with st.spinner("Analyzing traffic flow..."):
                data = {
                    'car_count': str(car_count),
                    'bike_count': str(bike_count),
                    'bus_count': str(bus_count),
                    'truck_count': str(truck_count)
                }
                
                result = call_api("/api/v1/traffic-flow/predict", data=data)
                
                if result:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Prediction Complete!")
                    
                    # Main metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        situation = result.get('predicted_situation', 'unknown')
                        situation_emoji = {
                            'light': '🟢',
                            'moderate': '🟡',
                            'heavy': '🔴'
                        }.get(situation.lower(), '⚪')
                        st.metric("Traffic Situation", f"{situation_emoji} {situation.title()}")
                    with col_b:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col_c:
                        congestion = result.get('congestion_level', 'unknown')
                        st.metric("Congestion Level", congestion.title())
                    
                    # Future projection
                    st.subheader("🔮 Future Projection (15 min ahead)")
                    predicted = result.get('predicted_vehicle_counts', {})
                    col_d, col_e, col_f, col_g = st.columns(4)
                    with col_d:
                        st.metric("Cars", predicted.get('car_count', 0))
                    with col_e:
                        st.metric("Bikes", predicted.get('bike_count', 0))
                    with col_f:
                        st.metric("Buses", predicted.get('bus_count', 0))
                    with col_g:
                        st.metric("Trucks", predicted.get('truck_count', 0))
                    
                    st.json(result)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Test Scenarios")
        st.markdown("""
        **Light Traffic:**
        - Cars: 5
        - Bikes: 2
        - Buses: 1
        - Trucks: 1
        
        **Moderate Traffic:**
        - Cars: 15
        - Bikes: 3
        - Buses: 2
        - Trucks: 5
        
        **Heavy Traffic:**
        - Cars: 25
        - Bikes: 5
        - Buses: 3
        - Trucks: 8
        """)


# Tab 4: Road Condition Detection
with tab4:
    st.header("🔧 Road Condition Detection")
    st.markdown("Detect potholes, cracks, and surface defects")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Road Surface Image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of road surface",
            key="condition_upload"
        )
        
        return_image = st.checkbox("Show processed image", value=True, key="condition_return_image")
        
        if st.button("🔍 Analyze Road", type="primary", key="condition_analyze_btn"):
            if uploaded_file is not None:
                with st.spinner("Analyzing road conditions..."):
                    files = {'file': uploaded_file.getvalue()}
                    data = {'return_image': str(return_image).lower()}
                    
                    result = call_api("/api/v1/road-condition/detect", files=files, data=data)
                    
                    if result:
                        if 'image' in result:
                            st.image(result['image'], caption="Processed Image with Road Condition Analysis", use_container_width=True)
                            result_data = result.get('data', {})
                        else:
                            result_data = result
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Analysis Complete!")
                        
                        # Summary metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Potholes", result_data.get('potholes_detected', 0))
                        with col_b:
                            st.metric("Cracks", result_data.get('cracks_detected', 0))
                        with col_c:
                            st.metric("Defects", result_data.get('surface_defects_detected', 0))
                        with col_d:
                            quality = result_data.get('overall_quality', 'unknown')
                            quality_emoji = {
                                'excellent': '🟢',
                                'good': '🟡',
                                'fair': '🟠',
                                'poor': '🔴',
                                'critical': '⚫'
                            }.get(quality.lower(), '⚪')
                            st.metric("Quality", f"{quality_emoji} {quality.title()}")
                        
                        # Quality details
                        col_e, col_f = st.columns(2)
                        with col_e:
                            st.metric("Quality Score", f"{result_data.get('quality_score', 0):.2%}")
                        with col_f:
                            priority = result_data.get('maintenance_priority', 'unknown')
                            st.metric("Maintenance Priority", priority.title())
                        
                        # Detailed results
                        if result_data.get('potholes'):
                            st.subheader("🕳️ Detected Potholes")
                            for i, pothole in enumerate(result_data['potholes'], 1):
                                with st.expander(f"Pothole {i}: {pothole.get('severity', 'unknown').title()} Severity"):
                                    st.json(pothole)
                        
                        if result_data.get('cracks'):
                            st.subheader("📏 Detected Cracks")
                            for i, crack in enumerate(result_data['cracks'], 1):
                                with st.expander(f"Crack {i}: {crack.get('type', 'unknown').title()} - {crack.get('severity', 'unknown').title()}"):
                                    st.json(crack)
                        
                        st.json(result_data)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please upload an image first")
        
        st.markdown("### 📸 Test Images")
        st.info("Try with: `data/test_images/lane.jpeg`, `lane2.jpeg`, `lane3.jpeg`")


# Tab 5: Complete Analysis
with tab5:
    st.header("🔄 Complete Image Analysis")
    st.markdown("Process image through all modules at once")
    
    uploaded_file = st.file_uploader(
        "Upload Road Scene Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a road scene image for complete analysis",
        key="complete_upload"
    )
    
    modules = st.multiselect(
        "Select Modules",
        ["road", "traffic_sign", "road_condition"],
        default=["road", "traffic_sign", "road_condition"],
        help="Choose which modules to run",
        key="complete_modules"
    )
    
    if st.button("🚀 Run Complete Analysis", type="primary", key="complete_analyze_btn"):
        if uploaded_file is not None:
            with st.spinner("Running complete analysis..."):
                files = {'file': uploaded_file.getvalue()}
                modules_str = ','.join(modules) if modules else 'all'
                data = {'modules': modules_str}
                
                result = call_api("/api/v1/process/image", files=files, data=data)
                
                if result:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("✅ Complete Analysis Finished!")
                    
                    results = result.get('results', {})
                    
                    # Display each module's results
                    if 'road_detection' in results:
                        with st.expander("🛣️ Road Detection Results", expanded=True):
                            rd = results['road_detection']
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Lane Detected", "✅ Yes" if rd.get('lane_detected') else "❌ No")
                            with col_b:
                                st.metric("Confidence", f"{rd.get('lane_confidence', 0):.2%}")
                            st.json(rd)
                    
                    if 'traffic_sign_detection' in results:
                        with st.expander("🚦 Traffic Sign Detection Results", expanded=True):
                            tsd = results['traffic_sign_detection']
                            st.metric("Signs Detected", tsd.get('signs_detected', 0))
                            if tsd.get('signs'):
                                for sign in tsd['signs']:
                                    st.write(f"**{sign.get('class_name', 'Unknown')}** - Confidence: {sign.get('confidence', 0):.2%}")
                            st.json(tsd)
                    
                    if 'road_condition_detection' in results:
                        with st.expander("🔧 Road Condition Detection Results", expanded=True):
                            rcd = results['road_condition_detection']
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Potholes", rcd.get('potholes_detected', 0))
                            with col_b:
                                st.metric("Cracks", rcd.get('cracks_detected', 0))
                            with col_c:
                                st.metric("Quality", rcd.get('overall_quality', 'unknown').title())
                            st.json(rcd)
                    
                    st.json(result)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please upload an image first")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Intelligent Traffic System API - Powered by FastAPI & Streamlit</p>
    <p>API Base URL: <code>{}</code></p>
</div>
""".format(API_BASE_URL), unsafe_allow_html=True)

