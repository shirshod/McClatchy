#!/usr/bin/env python
# coding: utf-8

# # Florida Corrections Department Inmate Mortality Scraper
# ###### By: Shirsho Dasgupta (2021)

# ### Notes: 
# ##### The code extracts each row from the inmate mortality tables, then goes into the corresponding inmate profile page and extracts the demographic and custody details. 
# ##### The code searches for the "td" tag which stores the details on both mortality and inmate profile tables.
# ##### Florida inmate mortality homepage: http://www.dc.state.fl.us/pub/mortality/index.html
# ##### Sample Florida inmate mortality URL for specific year: http://www.dc.state.fl.us/pub/mortality/2020-2021.html
# ##### Sample Florida inmate profile URL: http://www.dc.state.fl.us/OffenderSearch/detail.aspx?Page=Detail&DCNumber=126380&TypeSearch=IR

# ### Importing library

# In[1]:


import requests
from bs4 import BeautifulSoup
import time


# ### Creating spreadsheet

# In[2]:


with open("Inmate_Deaths.csv", "w") as f:
    f.write("fdc_number" + "\t" + "fdc_profile" + "\t" + "name" + "\t" + "race" + "\t" + "sex" + "\t" + "birth_date" + "\t" + "custody_type" + "\t" + "death_date" + "\t" + "death_year" + "\t" + "death_month_year" + "\t" + "facility" + "\t" + "death_manner" + "\t" + "case_status" + "\n")


# ### Extracting data and writing into spreadsheet

# In[3]:


# sets start and end years
start_date = 2016
end_date = 2021

# sets up loop to move through all relevant URLs (refer to notes)
for i in range(start_date, (end_date + 1), 1):
    
    # converts start and end years as strings 
    year_1 = str(i)
    year_2 = str(i+1)
    
    # going into mortality data page and converting it into readable form 
    url = "http://www.dc.state.fl.us/pub/mortality/" + year_1 + "-" + year_2 + ".html"
    page = requests.get(url)
    soup = BeautifulSoup(page.text)
    tables = soup.findAll('td')
    
    # sets up loop to move through each row and read each 'td' tag (refer to notes)
    counter = 0
    for table in tables:
        
        # exception handling to detect when the complete mortality table has been read
        try:
            
            #### THIS SECTION READS FROM THE INMATE MORTALITY PAGE
            ### extracting inmate number and URL of inmate profile page
            fdc_number = tables[counter + 1].text
            fdc_profile = "http://www.dc.state.fl.us/OffenderSearch/detail.aspx?Page=Detail&DCNumber=" + fdc_number + "&TypeSearch=IR"
            ### REVERTS TO INMATE PROFILE PAGE AFTER THIS POINT
            
            #### THIS SECTION READS THE INMATE PROFILE PAGE 
            ### going into inmate profile page and converting it into readable form
            profile_page = requests.get(fdc_profile)
            profile_soup = BeautifulSoup(profile_page.text)
            profile_table = profile_soup.findAll('td')
            
            ### extracting and storing inmate's name, race and sex
            name = profile_table[1].text
            race = profile_table[2].text
            sex = profile_table[3].text
            
            ### extracting date of birth
            birth_date_initial = profile_table[4].text
            
            ### storing date of birth in YYYY-MM-DD format
            birth_year = birth_date_initial[6:]
            birth_month = birth_date_initial[:2]
            birth_day = birth_date_initial[3:][:2]
            birth_date_final = birth_year + "-" + birth_month + "-" + birth_day
            
            ### extracting and storing type of custody 
            custody_type = profile_table[5].text
            #### END OF READING INMATE PROFILE PAGE
            
            #### REVERTS BACK TO INMATE MORTALITY PAGE
            ### extracting date of death
            death_date_initial = tables[counter + 2].text
            
            ### storing date of death in YYYY-MM-DD format
            death_year = death_date_initial[6:]
            death_month = death_date_initial[:2]
            death_day = death_date_initial[3:][:2]
            death_date_final = death_year + "-" + death_month + "-" + death_day
            death_month_final = death_year + "-" + death_month

            ### extracting and storing facility, manner of death and case status
            facility = tables[counter + 3].text
            death_manner = tables[counter + 4].text
            case_status = tables[counter + 5].text
            #### END of READING INMATE MORTALITY PAGE
            
            # counter reset to move onto next row
            counter = counter + 6

            ## writing details into spreadsheet
            with open("Inmate_Deaths.csv", "a") as f:
                f.write(fdc_number + "\t" + fdc_profile + "\t" + name + "\t" + race + "\t" 
                    + sex + "\t" + birth_date_final + "\t" + custody_type 
                    + "\t" + death_date_final + "\t" + death_year + "\t" + death_month_final + "\t"
                        + facility + "\t" + death_manner + "\t" + case_status + "\n")

        # moves onto next mortality data URL in case of exception
        except:
            i = i + 1
        
        # pause after writing each row to ensure server does not crash or access is not blocked off 
        time.sleep(0.01)


# In[ ]:




