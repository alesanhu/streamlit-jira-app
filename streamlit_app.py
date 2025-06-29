─── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:128 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:669 in code_to_exec                                     

                                                                                

  /mount/src/streamlit-jira-app/streamlit_app.py:208 in <module>                

                                                                                

    205                                                                         

    206                                                                         

    207 if __name__ == "__main__":                                              

  ❱ 208 │   main()                                                              

    209                                                                         

                                                                                

  /mount/src/streamlit-jira-app/streamlit_app.py:151 in main                    

                                                                                

    148 │   statuses   = sorted(df["status"].unique())                          

    149 │   priorities = sorted(df["priority"].unique())                        

    150 │   assignees  = sorted(df["assignee"].unique())                        

  ❱ 151 │   areas      = sorted(df["area_destino"].unique())                    

    152 │                                                                       

    153 │   sel_status = st.sidebar.multiselect("Estados", statuses, statuses)  

    154 │   sel_pri    = st.sidebar.multiselect("Prioridades", priorities, pri  

────────────────────────────────────────────────────────────────────────────────

TypeError: '<' not supported between instances of 'float' and 'str'
