// -----// IR Dump After LayoutPropagation (layout-propagation) //----- //
module {
  func.func @matvec(%arg0: !secret.secret<tensor<16xf32>> {tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>}) -> (!secret.secret<tensor<16xf32>> {tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %cst_0 = arith.constant dense<"0x5036CB3DED693F3E647E8E3F1C7029BFBEE1923F83F258BE77DDA1BFD7EDB93C309239C0842D833EB0C3163FD1DB2ABF5D971B3F1091853EED900CBE19464BBFE147C3BE584F72BF27116D3DAA05383F32109FBFC956703FADE2DCBEEE8F443E3EF9903F90A07CBF7803133FCDB2E83EDC29103FCC190A3FECD986BFAE4E853DE4A9393E3CB9E83EBA00FA3D0DBFF0BE79DF193F4E00073F29DA10BE1C9693BD12360BBFEEFCFBBD051EBDBEAE500E3F598190BF369C0D3F65AB74BF022E55BE47C021BEBD0E8D3EDDD4C93EBF82CB3F726237BFE1645E3FA4569FBDB0863BBFD15A1FC097BE063E94A451BFE4AA0B3F0C8726BEA2AD41BE1A15F63E1CDB073F40F376BF4D87BDBEA96AA03E8D9F843DBF6FFB3F9C8203BF24B92ABE7F70C5BE733F6C3F7CE7DA3E1F83C13F0284113EF339193F1B19E13CA1F5FA3E6E31C9BEA1078D3E5A0439BF1FD4A7BE3640A63F55B69FBF6D8B66BD4DF072BFC166A93E8D4BDF3FBF4AEABD9FD976BFB809763ECD4C10BFC88265BF6B2BABBEC202C5BE8D53EB3DBE94AABE3C3297BD75BF7FBE1EFEF83E1936893FAE8AA13EAF6CACBD615C2DC0473F593FE32D25BFDF71D0BD382D6CBE2DD392BD9C4FECB84BF853BED6E0493ECDCA91BE387D02BFF615053D7D4EB63F0042113E4C661B3FBF5F023F83868D3F497814BD911CAB3F4BC424BE7A020C3F6509A73E95E499BEEE54DB3EEFFC3CBF695FA93EA695923E1937CB3F553A37BF6EC745BF3DCF823EEC98153FC2D49C3F8523E03D07ED413D13E9853E016DA2BED73A6F3E2D0268BEEBC9613EBEB947BE870B93BE3402CB3EC68B41BE50F054BF161EB23E4FDC1F3EB562A1BDD9A115C0CCA8D6BDD65AFDBED757033F590DF63EC4280CBF8EA7EABE74C317BE4597B5BB576920BF6A4E0F3EEC66B9BE4072EE3F570AFE3D961D7C3E5FEE0ABD1EC09EBFEA77253E532AA23F15A656BE1923163DC24284BE374FFD3DB9F2A3BED185903E6294083F8D0700BFD998223FACE7A1BF79CC633C1039C43D8FB21CBF36CD1B3F99D43DBE85E451BF522F40BE8B94383D76727CBEADBB19BF755B6F3F9B9C1BBE4C08633D195E3E3E90944FBD511F9DBF8E44723F665C36BFCE54193F9ABE19C0ECACB53E88925B3EA19AA4BE1AD4EABEC5DE023FC759C83F37CE383FEB0713BDCBACC6BDCA2E0EBF28F39CBCE8A4C23F8D844E3F2791AF3DF31C79BE5129963F74A657BF3D09BD3DF1F9953EFDA50DBB79B7B8BE4A69D73EF01E2F3D23B418BFD8C9243F6CAA17BE21AE853F3C11793FBE51FABE47B452BE2A0A763E4CA4BDBF7AE739BE6C0AC33F0FDF2E3DBDA0BABC6E8E23BF37B836BF989532BE66736C3EABF141BE63FE853E26F7803F68822ABF7BFA0F3F34128B3EA3B655BFAB2F1C3ECC272F3F2B3A19BF198BD9BEA75C0DBF12C739BDE5F4E1BE1C591EBE"> : tensor<16x16xf32>
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<16xf32>>) attrs = {__argattrs = [{tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>}], tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>} {
    ^body(%input0: tensor<16xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #tensor_ext.layout<map = (d0, d1) -> (d0 * 16 + d1)>, tensor_ext.layout = #tensor_ext.layout<map = (d0, d1) -> (d0 * 16 + d1)>} : tensor<16x16xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #tensor_ext.layout<map = (d0) -> (d0)>, tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>} : tensor<16xf32>
      %3 = linalg.matvec {tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>} ins(%1, %input0 : tensor<16x16xf32>, tensor<16xf32>) outs(%2 : tensor<16xf32>) -> tensor<16xf32>
      secret.yield %3 : tensor<16xf32>
    } -> !secret.secret<tensor<16xf32>>
    return %0 : !secret.secret<tensor<16xf32>>
  }
}
