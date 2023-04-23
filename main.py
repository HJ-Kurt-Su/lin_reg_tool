import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import numpy as np
# import io
import statsmodels.formula.api as smf



@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

def ols_reg(formula, df):

  model = smf.ols(formula, df)
  res = model.fit()
  df_result = df.copy()
  df_result['yhat'] = res.fittedvalues
  df_result['resid'] = res.resid

#   print(df_result.head())

  return res, df_result, model


st.title('Linear Regression Tool')


# Provide dataframe example & relative url
data_ex_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTnqEkuIqYHm1eDDF-wHHyQ-Jm_cvmJuyBT4otEFt0ZE0A6FEQyg1tqpWTU0PXIFA_jYRX8O-G6SzU8/pub?gid=0&single=true&output=csv"
# st.write("Factor Format Example File [link](%s)" % factor_ex_url)
st.markdown("#### Data Format Example File [Demo File](%s)" % data_ex_url)

uploaded_csv = st.file_uploader('#### 選擇您要上傳的CSV檔')

if uploaded_csv is not None:
    df_raw = pd.read_csv(uploaded_csv, encoding="utf-8")
    st.header('您所上傳的CSV檔內容：')
    st.dataframe(df_raw)
    # fac_n = df_fac.shape[1]

    select_list = list(df_raw.columns)
    # select_list
    response = st.selectbox("### Choose Response(y)", select_list)
    # response
    factor_list = select_list.copy()
    factor_list.remove(response)
    factor = st.multiselect(
        "### Choose Factor(x)", factor_list)
    if not factor:
        st.error("Please select at least one factor.")
    # factor
    if st.button('Perform Analysis'):
        x_formula = "+".join(factor)
        formula = response + "~" + x_formula
        df_reg = df_raw.copy()
        # x_formula
        # formula
        # filter_list = factor + list(response)
        # filter_list
        # df_x = df_raw[factor]
        # df_x

        result, df_result, model = ols_reg(formula, df_reg)
        st.write(result.summary())

        csv = convert_df(df_result)
        date = str(datetime.datetime.now()).split(" ")[0]
        # table_filename = doe_type + "_table_" + date
        result_file = date + "_lin-reg.csv"
        st.download_button(label='Download DOE table as CSV', 
                            data=csv, 
                            file_name=result_file,
                            mime='text/csv')


    







#     doe_type = st.selectbox(
#     'Choose DoE Type:',
#     ("2 lv full", "response surface", "taguchi", "gsd", "latin-hypercube"))

# # Define doe type & show relative parameter based on DOE type
#     if doe_type == "taguchi":
#       taguchi_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQMX31nAb2tY_U33hGK-Y3Up-ta7sl55grC591QmS--kqz9EhQyCvxYMe9_fG3YnPIoZuiPlkMQ-zg_/pubhtml"
#       taguchi_dict = {"L4_3F2L":0, "L8_7F2L": 1, "L9_4F3L": 2, "L12_11F2L": 3, "L16_15F2L": 4, "L16b_5F4L": 5, "L18_1F2L-7F3L": 6}
#       st.markdown("#### Taguchi Array [Link](%s)" % taguchi_url)
      
#       taguchi_array = st.selectbox(
#         'Choose Taguchi Array:',
#         ("L4_3F2L", "L8_7F2L", "L9_4F3L", "L12_11F2L", "L16_15F2L", "L16b_5F4L", "L18_1F2L-7F3L"))
    

#     elif doe_type == "latin-hypercube":
#       lhc_criteria = st.selectbox(
#         'Choose Sample Method:',
#         (None, "center", "maximin", "centermaximin", "correlation"))

#       design_samples = st.number_input('Insert a number', min_value=1, value=10)

#     elif doe_type == "gsd":
#       gsd_url = "https://doi.org/10.1021/acs.analchem.7b00506"
#       st.markdown("#### GSD Introduction [Link](%s)" % gsd_url)

#       #@markdown Number of factor levels per factor in design. Must match factor q'ty  

#       #@markdown [3, 4] means 2 factor. 1st factor is 3 levels, 2nd factor is 4 levels.

#       # levels =  #@param {type:"raw"}

#       st.markdown("###### Define Level for Each Factor:")
#       levels=list()
#       a=0
#       for i in df_fac.columns:
#         tmp_lv = st.number_input(i, min_value=2, value=3, step=1, key=a)
#         levels.append(tmp_lv)
#         a+=1
#       # levels = [3, 4, 2, 3]

#       # print(len(levels))
#       # if doe_type == "gsd" and len(levels) != df_fac.shape[1]:
#       #   # print("--------------------------")
#       #   st.markdown("Factor Q'ty is not match, please check")
#       #   # print("--------------------------")
#       st.markdown("--------------------------")
#       st.markdown("###### Define GSD Rest Parameter:")
#       #@markdown Reduce the number of experiments to approximately half (Default 2).
#       reduction = st.number_input('Reduction', min_value=2, value=4, max_value=8, step=1)   
#       #@markdown Fold like to divide doe array (Default 1)
#       complementary = st.number_input('Complementary', min_value=1, max_value=4, step=1)
#       # complementary = 1 #@param {type:"slider", min:1, max:4, step:1}
#       #@markdown Choose which fold matrix (must < **complementary**)
#       select_fold_no = st.number_input('Select fold number', min_value=0, max_value=complementary-1, step=1)


#     st.write('You selected:', doe_type)

#     if st.button('Generate DOE Table'):
#       # st.write('Why hello there')

#       if doe_type == "response surface":

#         doe_array = pyDOE2.ccdesign(fac_n)
#         df_code = pd.DataFrame(doe_array)


#       elif doe_type == "2 lv full":
    
#         doe_array = pyDOE2.ff2n(fac_n)
#         df_code = pd.DataFrame(doe_array)


#       elif doe_type == "taguchi":

#         table_location = taguchi_dict[taguchi_array]
#         start_col = 2
#         df_tmp = pd.read_html(taguchi_url)[table_location]
#         df_code = df_tmp.iloc[1:,start_col:]
#         # print(df)
#         if fac_n > df_code.shape[1]:

#           # st.write('Most objects')
#           st.markdown("### Factor Q'ty is not enough, please select another array!!")

#         elif fac_n <= df_code.shape[1]:
#           df_code = df_code.iloc[:,:fac_n]
#           # fac_n
#           # df_code
      

#       elif doe_type == "latin-hypercube":
#         # lhc_criteria = para_dict["lhc_criteria"]
#         # design_samples = para_dict["design_samples"]

#         doe_lhs_array = pyDOE2.lhs(fac_n, samples=int(design_samples), criterion=lhc_criteria)
#         df_code = pd.DataFrame(doe_lhs_array)
      
#       elif doe_type == "gsd":
#         doe_gsd_arry = pyDOE2.gsd(levels, reduction, n=complementary)
#       # print(doe_gsd_arry)
#         if complementary == 1:
#           df_code = pd.DataFrame(doe_gsd_arry)
#         else:
#           df_code = pd.DataFrame(doe_gsd_arry[select_fold_no])


# # Turn DOE code table to mapping real factor upper & lower limit
#       df_resp = lv_doe_table(df_code, df_fac)
#       df_resp
          
#       # df = pd.read_csv("idc_nb_tidy.csv", encoding="utf-8")  
#       fig_pair = px.scatter_matrix(df_resp, dimensions=df_resp.columns,  
#                               width=640, height=480)
      
#       fig_pair.update_traces(diagonal_visible=False, showupperhalf=False,)
#       st.plotly_chart(fig_pair, use_container_width=True)

#       mybuff = io.StringIO()
#       fig_file_name = doe_type + "_pair-plot-test.html"
#       # fig_html = fig_pair.write_html(fig_file_name)
#       fig_pair.write_html(mybuff, include_plotlyjs='cdn')
#       html_bytes = mybuff.getvalue().encode()
      

#       csv = convert_df(df_resp)
#       doe_table = doe_type + "_doe-table.csv"
#       st.download_button(label='Download DOE table as CSV', 
#                         data=csv, 
#                         file_name=doe_table,
#                         mime='text/csv')

#       # st.download_button(label="Download figure",
#       #                           data=html_bytes,
#       #                           file_name=fig_file_name,
#       #                           mime='text/html'
#       #                           )