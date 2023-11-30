import matplotlib.pyplot
import easypyplot as epp

from plot_utils import *

# L1: conv_io64
# L7: conv_i64_o128
# L8: conv_io128
# L9: residual_i64_o128
# L16: conv_i128_o256
# L17: conv_io256
# L18: residual_i128_o256
# L29: conv_i256_o512
# L30: conv_io512
# L31: residual_i256_o512


if __name__ == '__main__':
    # ResNet-34 individual layers
    entries = [
        {
            'name': 'TVM',
            'data': [
                ('L1', 0.0017705682828282829),
                ('L7', 0.003364025995689655), ('L8', 0.0016910826248164466), ('L9', 0.00010187623215716033),
                ('L16', 0.0011380328052757796), ('L17', 0.002133188249094203), ('L18', 0.00010291491692373607),
                ('L29', 0.0013669781157167533), ('L30', 0.0028049316946386943), ('L31', 0.0001084908702228224),
            ],
            'baseline': True
        },
        {
            'name': 'Seq.1',
            'data': [
                ('L1', 0.00036160427807486626),
                ('L7', 0.00017739293501636277), ('L8', 0.00033674043000773393), ('L9', 0.00017692033115468409),
                ('L16', 0.00021598845140952894), ('L17', 0.0003739265565749236), ('L18', 0.00019915774122666668),
                ('L29', 0.00025121760289330925), ('L30', 0.0004902103494000407), ('L31', 0.0002515004225352113),
            ],
            'baseline': False
        },
        {
            'name': 'Seq.2',
            'data': [
                ('L1', 0.0008571687916666667),
                ('L7', 0.0003003373173173173), ('L8', 0.0006497032445278299), ('L9', 0.0003068630175736961),
                ('L16', 0.00038296540981516886), ('L17', 0.0006819508851490995), ('L18', 0.0003197398735927034),
                ('L29', 0.0003948253235300741), ('L30', 0.0007940586790697676), ('L31', 0.00037705052451922236),
            ],
            'baseline': False
        },
        {
            'name': 'Seq.3',
            'data': [
                ('L1', 0.00032970615703137437),
                ('L7', 0.00022470114572864324), ('L8', 0.0003689836801923655), ('L9', 0.00023088338296065665),
                ('L16', 0.0003649959520383693), ('L17', 0.0006909495479711123), ('L18', 0.0003674151151937984),
                ('L29', float('inf')), ('L30', float('inf')), ('L31', float('inf')),
            ],
            'baseline': False
        },
        {
            'name': 'Kernel 1',
            'data': [
                ('L1', 0.0001466349793494152),
                ('L7', 0.000382243529178338), ('L8', 0.00017417471419299004), ('L9', 0.000382243529178338),
                ('L16', 0.00047612942091262533), ('L17', 0.00047580105580693815), ('L18', 0.00047612942091262533),
                ('L29', 0.0010742284115226337), ('L30', 0.0010959971531400966), ('L31', 0.0010742284115226337),
            ],
            'baseline': False,
            'text_mark': True, 
            'offset': [(-0.18, 0.15), (0, 6.5), (-0.15, 0.15), (0, 0.70), (-0.1, 1), (-0.1, 0.30), (0, 0.40), (0, 2), (-0.03, 1), (0, 0.20)]
        },
        {
            'name': 'Kernel 2',
            'data': [
                ('L1', 4.806970422111158e-05),
                ('L7', 9.464780534650361e-05), ('L8', 6.59378067220052e-05), ('L9', 9.464780534650361e-05),
                ('L16', 0.00014507903373335722), ('L17', 0.0001732762502853881), ('L18', 0.00014507903373335722),
                ('L29', 0.00026306778587848937), ('L30', 0.00024370392659993934), ('L31', 0.00026306778587848937),
            ],
            'baseline': False,
            'text_mark': True, 
            'offset': [(0, 0.15), (0, 0.15), (0, 0.15), (0, 2.5), (0, 0.15), (0, 0.15), (0, 2), (0, 0.15), (0, 0.15), (0, 2)]
        },
    ]

    # Configurations
    name = 'kernel-performance'
    width = 0.7
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.6

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)
    num_groups = len(names)

    # Figures
    pp, fig = epp.pdf.plot_setup(f'analysis/results/{name}.pdf',
                                 font='default', figsize=(10.2, 2.4))
    ax = fig.gca()

    # Draw bars
    epp.barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      colors=[ansor_color, seq1_color, seq2_color, seq3_color, micro_nas_color, micro_nas_compress_color],
                      entry_names=names,
                      xticklabelfontsize=10,
                      breakdown=False)

    # Mark numbers
    text_numbers_custom(ax, width, entries, bars, fontsize=8)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylabel('Speedup ×', multialignment='center', fontsize=10)
    ax.set_ylim(0, 40)

    # Finish
    fig.tight_layout()
    fig.show()
    epp.pdf.plot_teardown(pp, fig)
