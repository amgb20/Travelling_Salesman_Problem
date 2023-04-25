from django import forms

class TSPForm(forms.Form):
    Lentgh = forms.IntegerField(min_value=1, max_value=100, label="Grid size X")
    Width = forms.IntegerField(min_value=1, max_value=100, label="Grid size Y")
    algorithm = forms.ChoiceField(choices=[('nn', 'Nearest Neighbor'), ('two-opt', '2-opt'), ('christofides', 'Christofides')])

