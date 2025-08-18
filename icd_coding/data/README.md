### 📁 ICD Coding data

This folder contains:
1. `ICD_Codes_and_Descriptions`: The strutured ICD-9-CM folder with all the available codes and their short and long descriptions. In total about ~13,000 codes. We used the long description column for making the FAISS index, for the codes in our dataset.
2. `ICD_Dataset_Sample`: A small sample of just 50 records used by the ICD Mapping Agent, each including the assigned ICD-9-CM code, its description (from `ICD_Codes_and_Descriptions`), the original and LLaMA 3.1 8B-standardized clinical notes. This is just for illustration.
3. `ACR_ICD_Mapping_Sample.xlsx`: A small sample for condition `Breast Pain` to show how the document is structured and the mapping being done by the `ACR_Criteria_Checker`
4. `index/icd_index.faiss`: FAISS index of the encoded icd code long descriptions.
5.  `index/icd_codes.json`: description and icd codes present in the dataset, for easier retrieval.

##### 📚 ICD-9-CM Code Ranges by Category

| Code Range  | Category                                                           |
| ----------- | ------------------------------------------------------------------ |
| 001–139     | Infectious and Parasitic Diseases                                  |
| 140–239     | Neoplasms                                                          |
| 240–279     | Endocrine, Nutritional, and Metabolic Diseases; Immunity Disorders |
| 280–289     | Diseases of the Blood and Blood-Forming Organs                     |
| 290–319     | Mental Disorders                                                   |
| 320–389     | Diseases of the Nervous System and Sense Organs                    |
| 390–459     | Diseases of the Circulatory System                                 |
| 460–519     | Diseases of the Respiratory System                                 |
| 520–579     | Diseases of the Digestive System                                   |
| 580–629     | Diseases of the Genitourinary System                               |
| 630–679     | Complications of Pregnancy, Childbirth, and the Puerperium         |
| 680–709     | Diseases of the Skin and Subcutaneous Tissue                       |
| 710–739     | Diseases of the Musculoskeletal System and Connective Tissue       |
| 740–759     | Congenital Anomalies                                               |
| 760–779     | Conditions Originating in the Perinatal Period                     |
| 780–799     | Symptoms, Signs, and Ill-Defined Conditions                        |
| 800–999     | Injury and Poisoning                                               |
| E & V Codes | External Causes of Injury and Supplemental Classification          |
