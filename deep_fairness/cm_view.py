from causalgraphicalmodels import CausalGraphicalModel as cm

model1 = cm(nodes=["Race","Gender","Transcript","View","Rating","Unknown_cause"], edges=[("Race", "Transcript"),("Race","View"),("Race","Rating"),("Gender","Transcript"),("Gender","View")])

model1.draw()